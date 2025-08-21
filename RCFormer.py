import torch
import torch.nn as nn
import math
from typing import Tuple, List


class PSDDEFeatureExtractor(nn.Module):
    def __init__(self, fs: int = 250, window_size: int = 250, stride: int = 50, feature_type: str = 'psd_de'):
        super().__init__()
        self.fs = fs
        self.window_size = window_size
        self.stride = stride
        
        if feature_type not in ['psd_de', 'psd', 'de']:
            raise ValueError("feature_type must be one of 'psd_de', 'psd', or 'de'")
        self.feature_type = feature_type

        # 定义EEG频段
        self.bands = [[1, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 30], [30, 45]]
        
        # 预计算每个频段对应的FFT索引
        self.freq_resolution = fs / window_size
        self.band_indices = []
        for low, high in self.bands:
            start_idx = int(low / self.freq_resolution)
            end_idx = int(high / self.freq_resolution)
            # 确保索引不越界
            if start_idx == end_idx:
                self.band_indices.append(torch.tensor([start_idx]))
            else:
                self.band_indices.append(torch.arange(start_idx, end_idx))
        
        self.register_buffer('hanning_window', torch.hann_window(self.window_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (b, c, length)
        # 1. 创建滑动窗口
        x = x.unfold(dimension=2, size=self.window_size, step=self.stride)
        # x 形状: (b, c, n_windows, window_size)
        
        x = x * self.hanning_window

        # 2. 对每个窗口进行FFT
        x_fft = torch.fft.rfft(x, n=self.window_size, dim=-1)
        power_spectrum = x_fft.abs()**2 / self.window_size

        # 3. 计算每个窗口在各频段的平均功率
        psd_features = []
        for indices in self.band_indices:
            indices = indices.to(x.device)
            avg_power = power_spectrum[..., indices].mean(dim=-1)
            psd_features.append(avg_power)
        
        psd_selected = torch.stack(psd_features, dim=-1) # (b, c, n_windows, num_bands)
        
        # 4. 计算DE并进行归一化
        epsilon = 1e-9
        de_selected = 0.5 * torch.log(2 * math.pi * math.e * psd_selected + epsilon)
        
        psd_norm = (psd_selected - psd_selected.mean(dim=(-2,-1), keepdim=True)) / (psd_selected.std(dim=(-2,-1), keepdim=True) + epsilon)
        de_norm = (de_selected - de_selected.mean(dim=(-2,-1), keepdim=True)) / (de_selected.std(dim=(-2,-1), keepdim=True) + epsilon)
        
        if self.feature_type == 'psd_de':
            features = torch.cat([psd_norm, de_norm], dim=-1)
        elif self.feature_type == 'psd':
            features = psd_norm
        else: # 'de'
            features = de_norm
        
        return features # 输出形状: (b, c, n_windows, 14)


class ClusterManager(nn.Module):
    def __init__(self, num_clusters: int, channels: int, update_rate: float = 0.2, distance_metric: str = 'euclidean'):
        super().__init__()
        self.num_clusters = num_clusters
        self.channels = channels
        self.update_rate = update_rate
        self.distance_metric = distance_metric
        
        self.register_buffer('pos_emb', None)
        self.register_buffer('centers', torch.zeros(num_clusters, 3))
        self.register_buffer('initialized', torch.tensor(False))
        
        self.fixed_cluster_sizes = self._calculate_fixed_cluster_sizes()

    def _calculate_fixed_cluster_sizes(self) -> torch.Tensor:
        base_size = self.channels // self.num_clusters
        remainder = self.channels % self.num_clusters
        sizes = [base_size + 1] * remainder + [base_size] * (self.num_clusters - remainder)
        return torch.tensor(sizes, dtype=torch.long)

    def _initialize_centers(self):
        pos = self.pos_emb
        indices = torch.zeros(self.num_clusters, dtype=torch.long, device=pos.device)
        dist_matrix = torch.cdist(pos, pos)
        start_node = torch.argmax(dist_matrix.sum(dim=1))
        indices[0] = start_node
        min_dists = dist_matrix[start_node].clone()
        for j in range(1, self.num_clusters):
            farthest_node = torch.argmax(min_dists)
            indices[j] = farthest_node
            min_dists = torch.minimum(min_dists, dist_matrix[farthest_node])
        self.centers = pos[indices]
        self.initialized.fill_(True)

    def _assign_channels_with_fixed_capacity(self, centers: torch.Tensor) -> torch.Tensor:
        pos = self.pos_emb
        dist_matrix = torch.cdist(pos, centers)
        sorted_dists, sorted_indices = torch.sort(dist_matrix, dim=1)
        assignments = torch.full((self.channels,), -1, dtype=torch.long, device=pos.device)
        cluster_counts = torch.zeros(self.num_clusters, dtype=torch.long, device=pos.device)
        for ch_idx in range(self.channels):
            for cluster_preference_idx in range(self.num_clusters):
                preferred_cluster = sorted_indices[ch_idx, cluster_preference_idx]
                if cluster_counts[preferred_cluster] < self.fixed_cluster_sizes[preferred_cluster]:
                    assignments[ch_idx] = preferred_cluster
                    cluster_counts[preferred_cluster] += 1
                    break
        return assignments

    def forward(self, features: torch.Tensor, pos_emb_batch: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.pos_emb = pos_emb_batch[0].clone()
            self._initialize_centers()
            
        temp_center_indices = self._find_farthest_channels(features)
        temp_assignments, _ = self._cluster_euclidean_temp(temp_center_indices, pos_emb_batch)
        
        batch_avg_temp_centers = torch.zeros_like(self.centers)
        for i in range(self.num_clusters):
            mask = (temp_assignments == i)
            if mask.sum() > 0:
                batch_avg_temp_centers[i] = pos_emb_batch.reshape(-1, 3)[mask.reshape(-1)].mean(dim=0)

        dist_matrix = torch.cdist(self.centers, batch_avg_temp_centers)
        matching = torch.argmin(dist_matrix, dim=1)
        matched_temp_centers = batch_avg_temp_centers[matching]
        self.centers = (1 - self.update_rate) * self.centers + self.update_rate * matched_temp_centers
        
        final_assignments = self._assign_channels_with_fixed_capacity(self.centers)
        return final_assignments
    
    def _find_farthest_channels(self, features: torch.Tensor) -> torch.Tensor:
        b, c, n_windows, d_feat = features.shape
        
        if self.distance_metric == 'riemannian':
            spd_matrices = torch.zeros(b, c, d_feat, d_feat, device=features.device)
            for i in range(b):
                for j in range(c):
                    channel_features = features[i, j, :, :].T
                    cov_matrix = torch.cov(channel_features)
                    spd_matrices[i, j] = cov_matrix + torch.eye(d_feat, device=features.device) * 1e-6
            
            
            def riemannian_distance(spd1, spd2):
                # 特征值分解实现逆平方根
                s, U = torch.linalg.eigh(spd1)
                s_sqrt_inv = torch.diag_embed(1.0/torch.sqrt(s.clamp(min=1e-8)))
                s1_inv_sqrt = U @ s_sqrt_inv @ U.mT
                
                # 计算几何均值矩阵
                g = s1_inv_sqrt @ spd2 @ s1_inv_sqrt.mT
                
                # 计算特征值对数距离
                log_eigs = torch.linalg.eigvalsh(g).clamp(min=1e-8).log()
                return log_eigs.square().sum(dim=-1).sqrt()
                
            dist_matrix = riemannian_distance(spd_matrices.unsqueeze(2), spd_matrices.unsqueeze(1))
        
        elif self.distance_metric == 'euclidean':
            features_flat = features.reshape(b, c, -1)
            dist_matrix = torch.cdist(features_flat, features_flat)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        selected_indices = torch.zeros(b, self.num_clusters, dtype=torch.long, device=features.device)
        for i in range(b):
            start_node = torch.argmax(dist_matrix[i].sum(dim=1))
            selected_indices[i, 0] = start_node
            min_dists = dist_matrix[i, start_node].clone()
            for j in range(1, self.num_clusters):
                farthest_node = torch.argmax(min_dists)
                selected_indices[i, j] = farthest_node
                min_dists = torch.minimum(min_dists, dist_matrix[i, farthest_node])
        return selected_indices

    def _cluster_euclidean_temp(self, centers_indices, pos_emb_b):
        b_idx = torch.arange(pos_emb_b.shape[0]).unsqueeze(1).to(centers_indices.device)
        center_coords = pos_emb_b[b_idx, centers_indices]
        dist = torch.cdist(pos_emb_b, center_coords)
        assignments = torch.argmin(dist, dim=-1)
        return assignments, None



class EEGRCformer(nn.Module):
    def __init__(self, channels: int, length: int, fs: int = 250, 
                 num_clusters: int = 5, d_model: int = 128, nhead: int = 4, 
                 num_encoder_layers: int = 3, dropout: float = 0.5,
                 feature_type: str = 'psd_de', 
                 window_size: int = 250, stride: int = 50,
                 use_clustering: bool = True,
                 distance_metric: str = 'riemannian'):
        super().__init__()
        
        self.use_clustering = use_clustering
        self.psd_de_extractor = PSDDEFeatureExtractor(fs, window_size, stride, feature_type)
        
        if self.use_clustering:
            self.cluster_manager = ClusterManager(num_clusters, channels, distance_metric=distance_metric)
        
        # 动态计算展平后的特征维度
        n_windows = (length - window_size) // stride + 1
        d_feat_band = 14 if feature_type == 'psd_de' else 7
        d_psdde_flat = n_windows * d_feat_band
        
        self.d_model = d_model
        self.projection_psdde = nn.Sequential(
            nn.Linear(d_psdde_flat, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.num_tokens = num_clusters if self.use_clustering else 1
        dim_feedforward = d_model * 2
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, capture_attention: bool = False):
        b, c, l = x.shape
        
        # 1. 提取特征 -> (B, C, n_windows, 14)
        features = self.psd_de_extractor(x)
        
        # 2. 展平并投影特征 -> (B, C, d_model)
        features_flat = features.reshape(b, c, -1)
        projected_features = self.projection_psdde(features_flat)
        
        # 3. 根据模式生成Tokens
        if self.use_clustering:
            # 使用原始特征进行聚类
            cluster_assignments = self.cluster_manager(features, pos_emb)
            tokens = torch.zeros(b, self.num_tokens, self.d_model, device=x.device)
            for j in range(self.num_tokens):
                mask = (cluster_assignments == j)
                if mask.sum().item() > 0:
                    cluster_ch_indices = torch.where(mask)[0]
                    # 对投影后的特征进行平均
                    tokens[:, j, :] = torch.mean(projected_features[:, cluster_ch_indices, :], dim=1)
        else:
            # 不分簇，对所有通道的投影后特征求平均
            tokens = torch.mean(projected_features, dim=1, keepdim=True)
            
        # 4. Transformer 编码
        src = tokens + self.pos_encoder
        encoded_tokens = self.transformer_encoder(src)
        
        output_features, _ = encoded_tokens.max(dim=1)
        return output_features
