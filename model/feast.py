import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

# --- Cache: Store KNN indices, coordinate arrays, and distance matrices per slide ---
# Key: hash value of barcode list, Value: (knn_indices, coords_array, spatial_distances) or (spatial_dist_matrix)
_knn_cache: dict = {}  # KNN related: (knn_indices, coords_array, spatial_distances)
_spatial_dist_cache: dict = {}  # For global attention: spatial_dist_matrix


def _get_barcode_hash(barcodes: list) -> str:
    """Convert barcode list to hash value to create cache key"""
    if barcodes is None:
        raise ValueError("barcodes cannot be None")
    # Convert barcode list to string and hash
    barcode_str = ','.join(sorted(barcodes))  # Sort to make order-independent
    return hashlib.md5(barcode_str.encode()).hexdigest()


def clear_spatial_cache():
    """Initialize spatial distance cache (for memory management)"""
    global _spatial_dist_cache
    _spatial_dist_cache.clear()


def clear_knn_cache():
    """Initialize KNN indices cache (for memory management)"""
    global _knn_cache
    _knn_cache.clear()


def clear_all_caches():
    """Initialize all caches (for memory management)"""
    clear_spatial_cache()
    clear_knn_cache()


def get_cache_info():
    """Return cache status information"""
    return {
        'knn_cache_size': len(_knn_cache),
        'spatial_dist_cache_size': len(_spatial_dist_cache),
    }


class LocalKNNFeastBlock(nn.Module):
    """
    Stage 1: Local KNN Attention Block
    Compute attention only with k nearest neighbors for all spots (original spots + pseudo spots)
    """
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 k_neighbors: int = 32, tau_neg: float = 0.6, beta: float = 1.5, 
                 delta: float = 0.6):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.k_neighbors = k_neighbors
        
        # Q, K, V projection
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # FFN (Feed-Forward Network)
        ffn_dim = int(feature_dim * 1.5)  # expansion ratio 1.5
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # LayerNorm for FFN
        self.ffn_norm = nn.LayerNorm(feature_dim)
        
        # Negative-aware attention hyperparameters
        self.tau_neg = tau_neg
        self.beta = beta
    
    def parse_barcode_to_coords(self, barcode):
        """Convert barcode string to coordinates (e.g., '20x10' -> (20, 10))"""
        try:
            parts = barcode.split('x')
            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
            else:
                return (0.0, 0.0)  # Default value
        except:
            return (0.0, 0.0)  # Default value on parsing failure
    
    def find_knn_indices(self, barcodes, device, dtype, use_cache=True):
        """
        Find k nearest neighbor indices for each spot and compute distances between neighbors.
        Caching support: compute once and reuse for the same barcode list.
        
        Returns:
            knn_indices: (N, k) index tensor
            coords_array: (N, 2) coordinate tensor
            spatial_distances: (N, k) distances between each query and k neighbors
        """
        if barcodes is None:
            raise ValueError("barcodes cannot be None")
        
        # Create cache key (include k_neighbors to use different cache for different k values)
        cache_key = None
        if use_cache:
            barcode_hash = _get_barcode_hash(barcodes)
            cache_key = f"{barcode_hash}_k{self.k_neighbors}"
        
        # Check cache
        if use_cache and cache_key is not None and cache_key in _knn_cache:
            cached_data = _knn_cache[cache_key]
            if len(cached_data) == 3:
                # New format: knn_indices, coords_array, spatial_distances
                cached_knn_indices, cached_coords, cached_spatial_dist = cached_data
            else:
                # Compatibility with old format: only knn_indices, coords_array
                cached_knn_indices, cached_coords = cached_data
                # Recompute spatial_distances
                query_coords = cached_coords.unsqueeze(1)  # (N, 1, 2)
                neighbor_coords = cached_coords[cached_knn_indices]  # (N, k, 2)
                cached_spatial_dist = torch.sqrt(
                    ((query_coords - neighbor_coords) ** 2).sum(dim=2)
                )  # (N, k)
                # Update cache
                _knn_cache[cache_key] = (cached_knn_indices, cached_coords, cached_spatial_dist)
            
            # Convert to match device and dtype
            knn_indices = cached_knn_indices.to(device=device)
            coords_array = cached_coords.to(device=device, dtype=dtype)
            spatial_distances = cached_spatial_dist.to(device=device, dtype=dtype)
            return knn_indices, coords_array, spatial_distances
        
        N = len(barcodes)
        
        # Extract coordinates
        coords_list = []
        for barcode in barcodes:
            coords = self.parse_barcode_to_coords(barcode)
            coords_list.append(coords)
        
        coords_array = torch.tensor(coords_list, dtype=dtype, device=device)  # (N, 2)
        
        # Compute distance matrix (N, N)
        diff = coords_array.unsqueeze(1) - coords_array.unsqueeze(0)  # (N, N, 2)
        distances = torch.sqrt((diff ** 2).sum(dim=2))  # (N, N)
        
        # Find k+1 nearest neighbors for each row (including self)
        k_actual = min(self.k_neighbors + 1, N)  # Include self
        _, knn_indices = torch.topk(distances, k=k_actual, dim=1, largest=False)  # (N, k+1)
        
        # Exclude the first one (self)
        knn_indices = knn_indices[:, 1:]  # (N, k)
        
        # Compute distances between k neighbors (pre-compute for caching)
        query_coords = coords_array.unsqueeze(1)  # (N, 1, 2)
        neighbor_coords = coords_array[knn_indices]  # (N, k, 2)
        spatial_distances = torch.sqrt(
            ((query_coords - neighbor_coords) ** 2).sum(dim=2)
        )  # (N, k)
        
        # Store in cache (save on CPU for memory efficiency)
        if use_cache and cache_key is not None:
            _knn_cache[cache_key] = (
                knn_indices.cpu(), 
                coords_array.cpu(), 
                spatial_distances.cpu()
            )
        
        return knn_indices, coords_array, spatial_distances
    
    def forward(self, features: torch.Tensor, barcodes: list = None, is_pseudo: torch.Tensor = None) -> torch.Tensor:
        N = features.shape[0]
        
        # Compute Q, K, V
        Q = self.q_proj(features)  # (N, feature_dim)
        K = self.k_proj(features)  # (N, feature_dim)
        V = self.v_proj(features)  # (N, feature_dim)
        
        # Reshape to multi-head
        Q = Q.view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)
        
        if barcodes is None:
            raise ValueError("barcodes cannot be None")
        
        tau_neg = self.tau_neg
        beta = self.beta
        
        # Find KNN indices (also return spatial_distances)
        knn_indices, _, spatial_distances = self.find_knn_indices(
            barcodes, features.device, features.dtype
        )  # knn_indices: (N, k), spatial_distances: (N, k)
        
        # Extract only k neighbor Keys and Values for each query
        # Use advanced indexing
        K_local = K[knn_indices]  # (N, k, num_heads, head_dim)
        V_local = V[knn_indices]  # (N, k, num_heads, head_dim)
        
        # Compute attention between Q and K_local
        # Q: (N, num_heads, head_dim), K_local: (N, k, num_heads, head_dim)
        # Scaled dot-product attention
        K_local_transposed = K_local.transpose(2, 3)  # (N, k, head_dim, num_heads)
        K_local_transposed = K_local_transposed.permute(0, 3, 1, 2)  # (N, num_heads, k, head_dim)
        
        raw_scores = torch.einsum('nhd,nhkd->nhk', Q, K_local_transposed) / (self.head_dim ** 0.5)  # (N, num_heads, k)
        
        # ALiBi penalty: use cached spatial_distances (no need to recompute)
        # spatial_distances is already computed and cached in find_knn_indices
        
        # Apply different scaling for each head
        head_scalings = torch.tensor(
            [1.0 / (2.0 ** (h + 1)) for h in range(self.num_heads)],
            device=features.device, dtype=features.dtype
        )  # (num_heads,)
        
        # Expand spatial_distances for each head
        spatial_distances_expanded = spatial_distances.unsqueeze(1)  # (N, 1, k)
        head_scalings_expanded = head_scalings.view(1, self.num_heads, 1)  # (1, num_heads, 1)
        scaled_spatial_dist = spatial_distances_expanded * head_scalings_expanded  # (N, num_heads, k)
        
        # Apply penalty to raw_scores
        raw_scores_pos = raw_scores - scaled_spatial_dist  # (N, num_heads, k)
        raw_scores_neg = -raw_scores - scaled_spatial_dist  # (N, num_heads, k)
        
        # Softmax normalization
        attention_weights = F.softmax(raw_scores_pos, dim=-1)  # (N, num_heads, k)
        attention_weights_negative = F.softmax(raw_scores_neg / tau_neg, dim=-1)  # (N, num_heads, k)
        
        attention_weights_final = attention_weights - beta * attention_weights_negative
        
        # Dropout
        attention_weights_final = self.dropout(attention_weights_final)
        
        # Apply attention
        # attention_weights_final: (N, num_heads, k), V_local: (N, k, num_heads, head_dim)
        V_local_transposed = V_local.permute(0, 2, 1, 3)  # (N, num_heads, k, head_dim)
        attended_features = torch.einsum('nhk,nhkd->nhd', attention_weights_final, V_local_transposed)  # (N, num_heads, head_dim)
        
        # Multi-head concat
        attended_features = attended_features.contiguous().view(N, self.feature_dim)  # (N, feature_dim)
        
        # Attention: Residual connection
        x = self.out_proj(attended_features) + features
        x = self.norm(x)
        
        # FFN: Feed-Forward Network
        ffn_output = self.ffn(x)
        final_output = ffn_output + x  # Residual connection
        final_output = self.ffn_norm(final_output)
        
        return final_output


class GlobalSelfFeastBlock(nn.Module):
    """
    Stage 2: Global Attention Block
    Compute attention for original spots only
    """
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1, 
                 tau_neg: float = 0.6, beta: float = 1.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Q, K, V projection
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(feature_dim)

        # FFN (Feed-Forward Network)
        ffn_dim = int(feature_dim * 1.5)  # expansion ratio 1.5
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # LayerNorm for FFN
        self.ffn_norm = nn.LayerNorm(feature_dim)

        # Negative-aware attention hyperparameters
        self.tau_neg = tau_neg
        self.beta = beta
    
    def parse_barcode_to_coords(self, barcode):
        """Convert barcode string to coordinates (e.g., '20x10' -> (20, 10))"""
        try:
            parts = barcode.split('x')
            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
            else:
                return (0.0, 0.0)  # Default value
        except:
            return (0.0, 0.0)  # Default value on parsing failure

    def compute_spatial_distance_matrix(self, barcodes, device='cpu', dtype=None, use_cache=True):
        """
        Compute spatial distance matrix from barcode list (within slide only) - vectorized version
        Caching support: compute once and reuse for the same barcode list
        """
        if barcodes is None:
            raise ValueError("barcodes cannot be None")
        
        # Create cache key
        cache_key = _get_barcode_hash(barcodes) if use_cache else None
        
        # Check cache
        if use_cache and cache_key is not None and cache_key in _spatial_dist_cache:
            cached_dist = _spatial_dist_cache[cache_key]
            # Convert to match device and dtype
            if cached_dist.device != device or cached_dist.dtype != dtype:
                cached_dist = cached_dist.to(device=device, dtype=dtype)
            return cached_dist
        
        N = len(barcodes)
        
        # Parse all barcodes at once (vectorized)
        coords_list = []
        for barcode in barcodes:
            coords = self.parse_barcode_to_coords(barcode)
            coords_list.append(coords)
        
        # Convert to numpy array then to tensor
        if dtype is None:
            dtype = torch.float
        coords_array = torch.tensor(coords_list, dtype=dtype, device=device)  # (N, 2)
        
        # Compute vectorized distances
        # coords_array: (N, 2), coords_array.unsqueeze(1): (N, 1, 2), coords_array.unsqueeze(0): (1, N, 2)
        diff = coords_array.unsqueeze(1) - coords_array.unsqueeze(0)  # (N, N, 2)
        distance_matrix = torch.sqrt((diff ** 2).sum(dim=2))  # (N, N)
        
        # Store in cache (save on CPU for memory efficiency)
        if use_cache and cache_key is not None:
            _spatial_dist_cache[cache_key] = distance_matrix.cpu()
        
        return distance_matrix
    
    def forward(self, features: torch.Tensor, barcodes: list = None, is_pseudo: torch.Tensor = None) -> torch.Tensor:

        N = features.shape[0]  # total_spots
        
        # 1. Compute Q, K, V
        Q = self.q_proj(features)  # (N, feature_dim)
        K = self.k_proj(features)  # (N, feature_dim)
        V = self.v_proj(features)  # (N, feature_dim)
        
        # Reshape to multi-head
        Q = Q.view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        K = K.view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        V = V.view(N, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)

        tau_neg = self.tau_neg
        beta = self.beta
        
        # 2. Compute attention (same logic as per-slide processing)
        attended_features = torch.zeros(N, self.num_heads, self.head_dim, device=features.device, dtype=features.dtype)
        
        # Process entire data as one slide (start_idx=0, end_idx=N)
        start_idx, end_idx = 0, N
        
        # Extract Q, K, V for current slide
        Q_slide = Q[start_idx:end_idx]  # (N, num_heads, head_dim)
        K_slide = K[start_idx:end_idx]  # (N, num_heads, head_dim)
        V_slide = V[start_idx:end_idx]  # (N, num_heads, head_dim)
        
        # Compute attention within slide only (Scaled Dot-Product Attention)
        raw_scores_slide = torch.einsum('nhd,mhd->nhm', Q_slide, K_slide) / (self.head_dim ** 0.5)
        
        # Apply spatial distance penalty (barcode-based, within slide only)
        if barcodes is None:
            raise ValueError("barcodes cannot be None")
        
        # Extract only barcodes for current slide
        slide_barcodes = barcodes[start_idx:end_idx]
        
        # Compute spatial distance matrix only within current slide
        slide_spatial_dist = self.compute_spatial_distance_matrix(
            slide_barcodes, device=features.device, dtype=features.dtype
        )
        
        # Apply different scaling for each head (1/2, 1/2^2, 1/2^3, ..., 1/2^num_heads)
        head_scalings = torch.tensor(
            [1.0 / (2.0 ** (h + 1)) for h in range(self.num_heads)], 
            device=features.device, dtype=features.dtype
        )  # (num_heads,)
        
        # Expand slide_spatial_dist with different scaling for each head
        slide_spatial_dist_expanded = slide_spatial_dist.unsqueeze(0)  # (1, N, N)
        head_scalings_expanded = head_scalings.view(1, self.num_heads, 1, 1)  # (1, num_heads, 1, 1)
        
        # Apply different scaling for each head
        scaled_spatial_dist = slide_spatial_dist_expanded.unsqueeze(1) * head_scalings_expanded  # (1, num_heads, N, N)
        scaled_spatial_dist = scaled_spatial_dist.squeeze(0)  # (num_heads, N, N)
        
        # Transpose raw_scores_slide from (N, num_heads, N) to (num_heads, N, N)
        raw_scores_slide_transposed = raw_scores_slide.transpose(0, 1)  # (num_heads, N, N)
        
        raw_scores_slide_pos = raw_scores_slide_transposed - scaled_spatial_dist
        raw_scores_slide_neg = - raw_scores_slide_transposed - scaled_spatial_dist
        
        # Softmax normalization
        attention_weights_slide = F.softmax(raw_scores_slide_pos, dim=-1)  # (num_heads, N, N)
        attention_weights_negative_slide = F.softmax(raw_scores_slide_neg / tau_neg, dim=-1)  # (num_heads, N, N)
        
        attention_weights_final_slide = attention_weights_slide - beta * attention_weights_negative_slide  # (num_heads, N, N)

        # Apply dropout
        attention_weights_final_slide = self.dropout(attention_weights_final_slide)
        
        # Apply attention to compute attended features
        # attention_weights_slide: (num_heads, N, N), V_slide: (N, num_heads, head_dim)
        # Transpose V_slide to (num_heads, N, head_dim)
        V_slide_transposed = V_slide.transpose(0, 1)  # (num_heads, N, head_dim)
        
        # Fix einsum: attention_weights_slide (num_heads, N, N) and V_slide_transposed (num_heads, N, head_dim)
        attended_features_slide = torch.einsum('nhm,nmd->nhd', attention_weights_final_slide, V_slide_transposed)  # (num_heads, N, head_dim)
        
        # Store result in full attended_features
        attended_features[start_idx:end_idx] = attended_features_slide.transpose(0, 1)  # (N, num_heads, head_dim)
        
        # Concat multi-head (already normalized per slide)
        attended_features = attended_features.contiguous().view(N, self.feature_dim)  # (N, feature_dim)
        
        # Attention: Residual connection
        x = self.out_proj(attended_features) + features
        x = self.norm(x)
        
        # FFN: Feed-Forward Network
        ffn_output = self.ffn(x)
        final_output = ffn_output + x  # Residual connection
        final_output = self.ffn_norm(final_output)
        
        return final_output


class TwoStageAttentionBlock(nn.Module):
    """
    Two-Stage Hierarchical Attention Block
    Stage 1: Local KNN Attention (original spots + pseudo spots) - O(N*k)
    Stage 2: Global Attention (original spots only) - O(N_original^2)
    """
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 k_neighbors: int = 32, tau_neg: float = 0.6, beta: float = 1.5):
        super().__init__()
        self.feature_dim = feature_dim
        
        # local attention block
        self.local_attention = LocalKNNFeastBlock(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            k_neighbors=k_neighbors,
            tau_neg=tau_neg,
            beta=beta
        )
        
        # global attention block
        self.global_attention = GlobalSelfFeastBlock(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            tau_neg=tau_neg,
            beta=beta
        )
    
    def forward(self, features: torch.Tensor, barcodes: list = None, is_pseudo: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: (N, feature_dim) - input features
            barcodes: list - barcode list
            is_pseudo: (N,) - 0 for real spots, 1 for pseudo spots
        """
        if barcodes is None:
            raise ValueError("barcodes cannot be None")
        
        # Stage 1: Local KNN Attention (process both real + pseudo)
        x = self.local_attention(features, barcodes=barcodes, is_pseudo=is_pseudo)
        
        # Stage 2: Global Attention (process only real spots)
        if is_pseudo is not None:
            # Extract and process only real spots
            real_indices = torch.where(is_pseudo == 0)[0]  # (N_real,)
            N_real = len(real_indices)
            
            if N_real > 0:
                # Extract only features and barcodes of real spots
                x_real = x[real_indices]  # (N_real, feature_dim)
                real_barcodes = [barcodes[i] for i in real_indices.cpu().tolist()]
                
                # Apply global attention (real spots only)
                x_real = self.global_attention(x_real, barcodes=real_barcodes, is_pseudo=None)
                
                # Place results back to original positions
                x[real_indices] = x_real
        else:
            # If is_pseudo is None, treat all spots as real
            x = self.global_attention(x, barcodes=barcodes, is_pseudo=None)
        
        return x


class FEAST(nn.Module):
    """
    FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics
    """
    def __init__(
        self, 
        input_dim: int, 
        num_blocks: int = 3,
        num_heads: int = 8, 
        dropout: float = 0.3, 
        num_genes: int = 250,
        k_neighbors: int = 32, 
        tau_neg: float = 0.6, 
        beta: float = 1.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.num_genes = num_genes
        
        # Input projection
        feature_dim = 1536 // 2
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim), 
            nn.ReLU(), 
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Two-Stage Hierarchical Attention Blocks (each block performs local -> global)
        self.attention_blocks = nn.ModuleList([
            TwoStageAttentionBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                k_neighbors=k_neighbors,
                tau_neg=tau_neg,
                beta=beta
            )
            for i in range(num_blocks)
        ])
        
        # MLP for gene expression prediction
        mlp_hidden_dim = feature_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_genes),
            nn.ReLU()
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # Initialize model parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(nn.LayerNorm, type(m)):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    def forward(self, features: torch.Tensor, barcodes: list = None, is_pseudo: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: (N, input_dim) - input features
            barcodes: list - barcode list
            is_pseudo: (N,) - 0 for real (original) spots, 1 for pseudo spots
        """
        if barcodes is None:
            raise ValueError("barcodes cannot be None")
        
        # Project input features
        x = self.input_projection(features)
        
        # Two-Stage Hierarchical Attention Blocks
        for block in self.attention_blocks:
            x = block(x, barcodes=barcodes, is_pseudo=is_pseudo)
        
        x = self.norm(x)
        
        # Filter only real (original) spots and pass to MLP for gene expression prediction
        if is_pseudo is not None:
            # Extract only real spots
            real_indices = torch.where(is_pseudo == 0)[0]  # (N_real,)
            if len(real_indices) > 0:
                x_real = x[real_indices]  # (N_real, feature_dim)
                # Predict gene expression with MLP (real spots only)
                gene_expression = self.mlp(x_real)  # (N_real, num_genes)
            else:
                # Return empty tensor if there are no real spots
                gene_expression = torch.empty(0, self.num_genes, 
                                             device=x.device, dtype=x.dtype)
        else:
            raise ValueError("is_pseudo is None")
        
        return gene_expression


