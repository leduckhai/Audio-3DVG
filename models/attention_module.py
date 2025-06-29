import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


class MinkowskiPointNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.projector = ME.MinkowskiLinear(in_channels, out_channels)
        self.pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, coords, feats):
        # coords: (N, 4), feats: (N, C)
        x = ME.SparseTensor(features=feats, coordinates=coords)
        x = self.projector(x)
        x = self.pool(x)  # Outputs a SparseTensor with shape (B, out_channels)
        return x.F  # Return dense feature: (B, out_channels)

def prepare_sparse_tensor(batch_points, voxel_size=0.05):
    """
    batch_points: Tensor of shape (B, N, 6)
    Returns:
        coords: (B*N', 4) [batch_idx, x, y, z]
        feats:  (B*N', C)
    """
    B, N, _ = batch_points.shape
    all_coords = []
    all_feats = []
    
    for b in range(B):
        pc = batch_points[b]  # (N, 6)
        xyz = pc[:, :3]
        feat = pc[:, 3:]

        # Voxelize coordinates
        coords = torch.floor(xyz / voxel_size).int()
        batch_idx = torch.full((coords.shape[0], 1), b, dtype=torch.int)
        coords_batched = torch.cat([batch_idx, coords], dim=1)  # (N, 4)

        all_coords.append(coords_batched)
        all_feats.append(feat)

    coords = torch.cat(all_coords, dim=0)  # (B*N, 4)
    feats = torch.cat(all_feats, dim=0)    # (B*N, C)
    return coords, feats

class AudioConditionedCrossAttention(nn.Module):
    def __init__(self, obj_dim, audio_dim, hidden_dim, num_heads=4):
        super(AudioConditionedCrossAttention, self).__init__()
        self.obj_dim = obj_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Object projections
        self.q_proj = nn.Linear(obj_dim, hidden_dim)
        self.k_proj = nn.Linear(obj_dim, hidden_dim)
        self.v_proj = nn.Linear(obj_dim, hidden_dim)

        # audio-conditioned bias terms
        self.q_audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.k_audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.v_audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, obj_dim)

    def forward(self, obj_feats, lang_embed):
        """
        obj_feats: (B, N, D)       - object features (batch, num_objects, obj_dim)
        lang_embed: (B, L)         - sentence embedding (batch, audio_dim)
        """
        B, N, _ = obj_feats.shape

        # Project audio to same dim as hidden_dim
        q_audio = self.q_audio_proj(lang_embed).unsqueeze(1)  # (B, 1, H)
        k_audio = self.k_audio_proj(lang_embed).unsqueeze(1)
        v_audio = self.v_audio_proj(lang_embed).unsqueeze(1)

        # Project objects
        Q = self.q_proj(obj_feats) + q_audio   # (B, N, H)
        K = self.k_proj(obj_feats) + k_audio
        V = self.v_proj(obj_feats) + v_audio

        # Reshape for multi-head attention
        def reshape(x):
            return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            # (B, num_heads, N, head_dim)

        Q = reshape(Q)
        K = reshape(K)
        V = reshape(V)

        # Scaled dot-product attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, heads, N, N)

        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Combine heads
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)  # (B, N, H)

        # Project back to object feature dim
        out = self.out_proj(attended)  # (B, N, obj_dim)

        return out  # audio-modulated object features


class TargetToRelationalCrossAttention(nn.Module):
    def __init__(self, target_dim, relational_dim, audio_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Projections
        self.q_proj = nn.Linear(target_dim, hidden_dim)
        self.k_proj = nn.Linear(relational_dim, hidden_dim)
        self.v_proj = nn.Linear(relational_dim, hidden_dim)

        # audio modulation
        self.q_audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.k_audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.v_audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, target_dim)

    def forward(self, targets, relationals, audio_feat):
        """
        targets:     (B, N, d_t) — N target candidate features
        relationals: (B, N, d_r) — N relational object features
        audio_feat:   (B, d_l)    — audio sentence embedding
        Returns:
            updated_targets: (B, N, d_t)
        """
        B, N, _ = targets.shape

        # Project audio
        q_audio = self.q_audio_proj(audio_feat).unsqueeze(1)  # (B, 1, H)
        k_audio = self.k_audio_proj(audio_feat).unsqueeze(1)
        v_audio = self.v_audio_proj(audio_feat).unsqueeze(1)

        # Project inputs
        Q = self.q_proj(targets) + q_audio  # (B, N, H)
        K = self.k_proj(relationals) + k_audio
        V = self.v_proj(relationals) + v_audio

        # Reshape for multi-head
        def reshape(x):
            return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape(Q)  # (B, heads, N, head_dim)
        K = reshape(K)
        V = reshape(V)

        # Attention: Q from targets, K/V from relational objects
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Merge heads
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)  # (B, N, H)

        # Final projection
        updated_targets = self.out_proj(attended)  # (B, N, d_t)

        return updated_targets


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
                    nn.Linear(128, 1)
                    )
        self.target_projector = nn.Sequential(
                    nn.Linear(312, 128)
                    )
        
        self.relation_projector = nn.Sequential(
                    nn.Linear(312, 128)
                    )
        self.audio_projector = nn.Sequential(
                    nn.Linear(1024, 512)
                    )
        self.Minkowski_model = MinkowskiPointNet(in_channels=3, out_channels=512)
        self.attn_layer = AudioConditionedCrossAttention(obj_dim=128, audio_dim=1024, hidden_dim=128)
        self.cross_attn = TargetToRelationalCrossAttention(128, 128, audio_dim=1024, hidden_dim=128)
    def forward(self, data_dict):

        pointcloud = data_dict["point_clouds"][:, :, :6]
        coords, feats = prepare_sparse_tensor(pointcloud)
        coords = coords.to('cuda')
        feats = feats.to('cuda')
        scene_embedding = self.Minkowski_model(coords, feats) # B x 512
        
        target_representation = data_dict["target_representation"]
        relation_representation = data_dict["relation_representation"]
        bts_audio_feature = data_dict["bts_audio_feature"].squeeze(1)
        bts_audio_feature = self.audio_projector(bts_audio_feature)

        target_representation = self.target_projector(target_representation)
        relation_representation = self.target_projector(relation_representation)

        attns = self.attn_layer(target_representation, torch.cat([bts_audio_feature, scene_embedding], dim=1))
        cross_attns = self.cross_attn(target_representation, relation_representation, torch.cat([bts_audio_feature, scene_embedding], dim=1))

        scores = self.fc(0.5*target_representation + 0.3*attns + 0.2*cross_attns)
        data_dict['score'] = scores
        return data_dict