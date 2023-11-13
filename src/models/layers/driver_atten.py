import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('.')
sys.path.append('..')
from src.models.layers import SelfAttentionLayer, clones

from math import factorial
from utils import global_dict_init, set_value


class DriverAttenModel(nn.Module):
    """Driver model"""

    def __init__(
        self,
        num_dense_features,
        num_sparse_features,
        embedding_dims,
    ):
        super(DriverAttenModel, self).__init__()
        self.num_dense_features = num_dense_features
        self.num_sparse_features = len(embedding_dims)
        self.embedding_dims = embedding_dims

        # Dense MLP
        dense_layers = [nn.Linear(1, embedding_dims[-1]), nn.ReLU()]
        self.dense_mlp = nn.Sequential(*dense_layers)

        # Sparse embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, d)
            for n, d in zip(num_sparse_features, embedding_dims)
        ])

        # self attention
        self.self_attention = clones(
            SelfAttentionLayer(head_num=4,
                               d_model=256,
                               d_ff=4 * 256,
                               norm_dim=106 if type == 'bn' else None,
                               dropout=0.2), 1)

    def forward(self, dense_features: torch.Tensor,
                sparse_features: torch.Tensor):
        """Drver Atten Forward Func

        Args:
            dense_features (torch.Tensor): Dense feature [B, F_dense]
            sparse_features (toorch.Tensor): sparse feature [B, F_sparse]

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        dense_features = dense_features.unsqueeze(-1)  # [B, num_dense, 1]
        dense_features = self.dense_mlp(
            dense_features)  # [B, num_dense, embedding_dim]
        sparse_embeds = [
            emb(sparse_features[:, i]) for i, emb in enumerate(self.embeddings)
        ]  # [len(sparse), B, embedding_dim]
        sparse_features = torch.stack(sparse_embeds,
                                      dim=1)  # [B, num_sparse, embeding_dim]
        input_features = torch.cat(
            [dense_features, sparse_features],
            dim=1).unsqueeze(2)  # [B, num_sparse+num_dense, 1, embedding_dim]
        # Self attention
        for atten in self.self_attention:
            input_features, atten_scores = atten(input_features, mask=None)
        features = input_features
        return features


if __name__ == "__main__":
    # fake input
    global_dict_init()
    set_value("norm_type", 'ln')
    num_dense_features = 13
    num_sparse_features = [10, 20, 30, 40, 50, 20]
    embedding_dims = [256, 256, 256, 256, 256, 256]
    mlp_dims = [64, 32, 16]
    interaction_type = "attention"
    attention_dim = 16
    model = DriverAttenModel(num_dense_features, num_sparse_features,
                             embedding_dims, mlp_dims, interaction_type,
                             attention_dim)
    dense_features = torch.randn(4, num_dense_features)  # [4, 13]
    sparse_features = torch.cat(
        [torch.randint(0, n, (4, 1)) for n in num_sparse_features],
        dim=1)  # [4, 5]
    out = model(dense_features, sparse_features)
    print(out)
