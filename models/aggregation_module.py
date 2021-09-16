import torch
import torch.nn as nn

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward

class ContextualAggregationModule(nn.Module):
    def __init__(self, num_proposals=256, hidden_size=128, head=4, num_locals=10):
        super().__init__()

        self.num_proposals = num_proposals
        self.hidden_size = hidden_size

        self.head = head
        self.num_locals = num_locals

        self.self_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)
        self.self_attn_ffn = self._get_ffn_with_zero_initialization()
        
        self.cross_attn = MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head)
        self.cross_attn_ffn = PositionWiseFeedForward(d_model=hidden_size)

    def _get_ffn_with_zero_initialization(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)

        ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3)
        )
        
        ffn.apply(init_weights)

        return ffn

    def forward(self, centers, features, o_centers, o_features):
        # self-attention
        self_dist_weights = self.compute_spatial_proximity(centers, centers)
        self_dist_weights = torch.cat([self_dist_weights for _ in range(self.head)], dim=1).detach()
        features = self.self_attn(features, features, features, attention_weights=self_dist_weights, way="add")
        offsets = self.self_attn_ffn(features) # B, K, 3
        immediate_centers = centers + offsets

        # cross attention
        cross_dist_weights = self.compute_spatial_proximity(immediate_centers, o_centers)
        cross_dist_weights = torch.cat([cross_dist_weights for _ in range(self.head)], dim=1).detach()
        features = self.cross_attn(features, o_features, o_features, attention_weights=cross_dist_weights, way="add")
        features = self.cross_attn_ffn(features)

        return immediate_centers, features

    def compute_spatial_proximity(self, src_centers, tar_centers):
        def k_nearest_neighbors(dist_matrix):
            '''
                dist_matrix: (B, K, K)
                masks: (B, K)    
            '''

            new_dist_matrix = dist_matrix.reshape(-1, self.num_proposals) # B * K, K
            _, topk_ids = torch.topk(new_dist_matrix, self.num_locals, largest=False, dim=1) # B * K, num_locals
            # construct masks for the local context
            local_masks = torch.zeros(topk_ids.shape[0], self.num_proposals).type_as(topk_ids)
            local_masks.scatter_(1, topk_ids, 1) # B * K, K

            # new_dist_matrix.masked_fill_(local_masks == 0, float('1e30')) # dist to objects outside the neighborhood: 1e30
            # new_dist_matrix = new_dist_matrix.reshape(-1, self.num_proposals, self.num_proposals) # B, K, K

            local_masks = local_masks.reshape(-1, self.num_proposals, self.num_proposals) # B, K, K

            return local_masks

        # Attention Weight
        N_K = src_centers.shape[1]
        center_A = src_centers[:, None, :, :].repeat(1, N_K, 1, 1)
        center_B = tar_centers[:, :, None, :].repeat(1, 1, N_K, 1)
        dist = (center_A - center_B).pow(2)
        # print(dist.shape, '<< dist shape', flush=True)
        dist = torch.sqrt(torch.sum(dist, dim=-1)) # B, K, K
        local_masks = k_nearest_neighbors(dist).unsqueeze(1) # B, 1, K, K
        dist = dist.unsqueeze(1) # B, 1, K, K
        dist_weights = 1 / (dist+1e-2)
        norm = torch.sum(dist_weights, dim=2, keepdim=True)
        dist_weights = dist_weights / norm # B, 1, K, K
        dist_weights.masked_fill_(local_masks == 0, float('-inf'))

        return dist_weights



