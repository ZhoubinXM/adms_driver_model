import torch
import torch.nn as nn
import torch.nn.functional as F

from math import factorial


class AttentionInteraction(nn.Module):

    def __init__(self, num_sparse_features, embedding_dim, attention_dim):
        super(AttentionInteraction, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_dim), nn.ReLU(),
            nn.Linear(attention_dim, 1))
        self.num_sparse_features = num_sparse_features
        self.embedding_dim = embedding_dim

    def forward(self, embeddings):
        batch_size = embeddings[0].shape[0]
        interactions = []
        for i in range(self.num_sparse_features):
            for j in range(i + 1, self.num_sparse_features):
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                interaction = torch.cat([emb_i, emb_j], dim=-1)
                interaction = self.attention_layer(interaction)
                interactions.append(interaction)
        interactions = torch.cat(interactions, dim=-1)
        return interactions


class DriverModel(nn.Module):
    """Driver model"""

    def __init__(self,
                 num_dense_features,
                 num_sparse_features,
                 embedding_dims,
                 mlp_dims,
                 interaction_type="dot",
                 attention_dim=16):
        super(DriverModel, self).__init__()
        self.num_dense_features = num_dense_features
        self.num_sparse_features = len(embedding_dims)
        self.embedding_dims = embedding_dims
        self.mlp_dims = mlp_dims
        self.interaction_type = interaction_type

        # Dense MLP
        dense_layers = [nn.Linear(num_dense_features, mlp_dims[0]), nn.ReLU()]
        for i in range(1, len(mlp_dims)):
            dense_layers.extend(
                [nn.Linear(mlp_dims[i - 1], mlp_dims[i]),
                 nn.ReLU()])
        self.dense_mlp = nn.Sequential(*dense_layers)

        # Sparse embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, d)
            for n, d in zip(num_sparse_features, embedding_dims)
        ])

        # Feature Interaction
        if interaction_type == "attention":
            self.interaction_layer = AttentionInteraction(
                len(num_sparse_features), embedding_dims[-1], attention_dim)
        elif interaction_type == 'mlp':
            self.interaction_layer = None
        # input_dim = mlp_dims[-1] + (self.num_sparse_features * (self.num_sparse_features - 1)) // 2 * sum(embedding_dims)
        # input_dim = mlp_dims[-1] + factorial(
        #     self.num_sparse_features) // (factorial(2) * factorial(self.num_sparse_features - 2))
        input_dim = mlp_dims[-1] + (self.num_sparse_features *
                                    (self.num_sparse_features - 1)) // 2
        # top_mlp_dims = [input_dim] + mlp_dims

        # Output MLP
        top_layers = []
        # for i in range(1, len(top_mlp_dims)):
        #     top_layers.extend(
        #         [nn.Linear(top_mlp_dims[i - 1], top_mlp_dims[i]),
        #          nn.ReLU()])
        top_layers.append(nn.Linear(input_dim, 256))
        self.top_mlp = nn.Sequential(*top_layers)

    def forward(self, dense_features, sparse_features):
        dense_out = self.dense_mlp(dense_features)
        sparse_embeds = [
            emb(sparse_features[:, i]) for i, emb in enumerate(self.embeddings)
        ] # [len(sparse), B, embedding_dim]
        if self.interaction_type == "dot":
            sparse_stack = torch.stack(sparse_embeds, dim=1)
            pairwise_inner_products_sparse = torch.matmul(
                sparse_stack, sparse_stack.transpose(-2, -1))
            triu_indices = torch.triu_indices(self.num_sparse_features,
                                              self.num_sparse_features, 1)
            interact = pairwise_inner_products_sparse[:, triu_indices[0],
                                                      triu_indices[1]]
            interact = interact.reshape(dense_out.shape[0], -1)
        elif self.interaction_type == "attention":
            interact = self.interaction_layer(sparse_embeds)
        elif self.interaction_type == "mlp":
            pass
        else:
            raise NotImplementedError(
                f"Interaction type '{self.interaction_type}' not implemented")
        features = torch.cat([dense_out, interact], dim=1)
        features = self.top_mlp(features)
        return features


if __name__ == "__main__":
    # fake input
    num_dense_features = 13
    num_sparse_features = [10, 20, 30, 40, 50, 20]
    embedding_dims = [16, 16, 16, 16, 16, 16]
    mlp_dims = [64, 32, 16]
    interaction_type = "attention"
    attention_dim = 16
    model = DriverModel(num_dense_features, num_sparse_features,
                        embedding_dims, mlp_dims, interaction_type,
                        attention_dim)
    dense_features = torch.randn(4, num_dense_features)  # [4, 13]
    sparse_features = torch.cat(
        [torch.randint(0, n, (4, 1)) for n in num_sparse_features],
        dim=1)  # [4, 5]
    out = model(dense_features, sparse_features)
    print(out)
