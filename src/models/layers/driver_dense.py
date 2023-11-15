import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import *


class DriverDenseModel(nn.Module):
    """Driver model"""

    def __init__(self, num_dense_features, num_sparse_features, embedding_dims, mlp_dims, attention_dim=16):
        super(DriverDenseModel, self).__init__()
        self.num_dense_features = num_dense_features
        self.num_sparse_features = len(embedding_dims)
        self.sparse_features = num_sparse_features
        self.embedding_dims = embedding_dims

        self.dense_mlp = MLP(in_features=num_dense_features + sum(self.sparse_features), out_features=256,
                             hidden_features=256, end_with_linear=True)

    def forward(self, dense_features, sparse_features: torch.Tensor):
        # sparse feature one-hot encoding
        B, F = sparse_features.shape
        batch_sparse = []
        for i in range(B):
            sparse_one_hot = []
            for j in range(F):
                feature_class = self.sparse_features[j]
                feature_one_hot = nn.functional.one_hot(sparse_features[i][j], feature_class)
                sparse_one_hot.append(feature_one_hot)
            batch_sparse.append(torch.cat(sparse_one_hot))
        batch_sparse = torch.stack(batch_sparse, dim=0)
        input_features = torch.cat([dense_features, batch_sparse], dim=-1)
        return self.dense_mlp(input_features)


if __name__ == "__main__":
    # fake input
    num_dense_features = 13
    num_sparse_features = [10, 20, 30, 40, 50, 20]
    embedding_dims = [16, 16, 16, 16, 16, 16]
    mlp_dims = [64, 32, 16]
    interaction_type = "attention"
    attention_dim = 16
    model = DriverModel(num_dense_features, num_sparse_features, embedding_dims, mlp_dims, interaction_type,
                        attention_dim)
    dense_features = torch.randn(4, num_dense_features)  # [4, 13]
    sparse_features = torch.cat([torch.randint(0, n, (4, 1)) for n in num_sparse_features], dim=1)  # [4, 5]
    out = model(dense_features, sparse_features)
    print(out)
