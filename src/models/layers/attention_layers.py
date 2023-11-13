import torch
import torch.nn as nn
import math
import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from src.models.layers.base import clones, Norm

from typing import Tuple, Optional


class MultiHeadAttention_0(nn.Module):

    def __init__(self, head_num: int, d_model: int, dropout=0.1) -> None:
        super(MultiHeadAttention_0, self).__init__()
        self.atten = torch.nn.MultiheadAttention(
            d_model,
            num_heads=head_num,
            batch_first=True,
        )
        self.head_num = head_num

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> torch.Tensor:
        """
            Args:
                query: [batch_size, num1, 1, dim]
                key: [batch_size, num2, 1, dim]
                value: [batch_size, num2, 1, dim]
                mask: [batch_size, 1, 1, num2]
        """
        # query, key, value: [batch_size, num, 1, dim]
        # mask: [batch_size, 1, 1, num]
        if mask is not None:
            mask = (mask - 1) * 1e9
            mask = mask.repeat(self.head_num, 1, query.size(1), 1)
            x, attention_scores = self.atten(query=query.squeeze(),
                                            key=key.squeeze(),
                                            value=value.squeeze(),
                                            attn_mask=mask.squeeze())
        else:
            x, attention_scores = self.atten(query=query.squeeze(),
                                            key=key.squeeze(),
                                            value=value.squeeze())
        return x.unsqueeze(2), attention_scores


class MultiHeadAttention(nn.Module):

    def __init__(self, head_num: int, d_model: int, dropout=0.1, type=None) -> None:
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % head_num == 0
        #  We assume d_v always equals d_k, d_k is head_size
        # That is d_k == d_v == d_model / head_num
        self.head_dim = d_model // head_num
        self.head_num = head_num
        self.linears = clones(nn.Linear(d_model, d_model, bias=True), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.type_ = type

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> torch.Tensor:
        """
            Args:
                query: [batch_size, num1, 1, dim]
                key: [batch_size, num2, 1, dim]
                value: [batch_size, num2, 1, dim]
                mask: [batch_size, 1, 1, num2]
        """

        # 1) Do all the linear projections in batch
        #    from [batch_size, num, 1, dim] to [batch_size, num, head_num, head_dim]
        query, key, value = [
            l(x).view(x.size()[0], -1, self.head_num, self.head_dim).transpose(1, 2) \
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        (x, attention_scores) = attention(query,
                                          key,
                                          value,
                                          mask=mask,
                                          dropout=self.dropout,
                                          type=self.type_)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(-3,
                        -2).contiguous().view(query.size()[0], -1, 1,
                                              self.head_dim * self.head_num)
        x = self.linears[-1](x)

        return x, attention_scores


class PointerwiseFeedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout=0.1) -> None:
        super(PointerwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class SelfAttentionLayer(nn.Module):
    """
    SelfAttentionLayer is made up of muti_attn and feed forward (defined below)
    """

    def __init__(self,
                 head_num: int,
                 d_model: int,
                 d_ff: int,
                 norm_dim: Optional[int] = None,
                 dropout: Optional[float] = 0.1) -> None:
        super(SelfAttentionLayer, self).__init__()
        self.muti_attn = MultiHeadAttention(head_num=head_num,
                                            d_model=d_model,
                                            dropout=dropout)
        self.feed_forward = PointerwiseFeedforward(d_model,
                                                   d_ff,
                                                   dropout=dropout)
        if norm_dim is None:
            norm_dim = d_model
        self.norm = clones(Norm(norm_dim), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Follow Figure 1 (left) for connections.
        """
        short_cut = x
        x, scores = self.muti_attn(x, x, x, mask)
        x = self.norm[0](x + short_cut)
        x = self.norm[1](x + self.feed_forward(x))
        return x, scores


class CrossAttentionLayer(nn.Module):
    """
    CrossAttentionLayer is made up of muti_attn and feed forward (defined below)
    """

    def __init__(self,
                 head_num: int,
                 d_model: int,
                 d_ff: int,
                 norm_dim: Optional[int] = None,
                 dropout: Optional[float] = 0.1, type=None) -> None:
        super(CrossAttentionLayer, self).__init__()
        self.muti_attn = MultiHeadAttention(head_num=head_num,
                                            d_model=d_model,
                                            dropout=dropout, type=type)
        self.feed_forward = PointerwiseFeedforward(d_model,
                                                   d_ff,
                                                   dropout=dropout)
        if norm_dim is None:
            norm_dim = d_model
        self.norm = clones(Norm(norm_dim), 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Follow Figure 1 (left) for connections.
        """
        short_cut = x
        x, scores = self.muti_attn(x, y, y, mask)
        x = self.norm[0](x + short_cut)
        x = self.norm[1](x + self.feed_forward(x))
        return x, scores
        # x = self.norm[0](x + self.muti_attn(x, y, y, mask))
        # x = self.norm[1](x + self.feed_forward(x))
        # return x


def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask=None,
              dropout=None, type=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 'Scaled Dot Product Attention'
    """
    head_dim = query.size(-1)
    # Q,K,V: [batch_size, num, head_num, head_dim]
    # scores: [batch_size, head_num, num1, num2]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    # mask: [batch_size, 1 ,1 ,num2] => dimension expansion
    if mask is not None:
        mask = (mask - 1) * 1e9
        scores = scores + mask
        # scores = scores.masked_fill_(mask == 0, value=-1e9)
    if type is not None:
        attention_scores = torch.softmax(scores, dim=-2)
    else:
        attention_scores = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attention_scores = dropout(attention_scores)
    return torch.matmul(attention_scores, value), attention_scores


if __name__ == "__main__":
    # set global dict
    from utils import global_dict_init, set_value
    global_dict_init()
    set_value("norm_type", 'ln')

    # Test MultiHeadAttention
    batch_size = 2
    num1 = 3
    num2 = 4
    head_num = 2
    d_model = 8
    query = torch.randn(batch_size, num1, 1, d_model)
    key = torch.randn(batch_size, num2, 1, d_model)
    value = torch.randn(batch_size, num2, 1, d_model)
    mask = torch.randn(batch_size, 1, 1, num2)
    torch_attn = torch.nn.MultiheadAttention(
        d_model,
        num_heads=head_num,
        # add_bias_kv=qkv_bias,
        # dropout=attn_drop,
        batch_first=True,
    )

    torch_x, torch_scores = torch_attn(
        query=query.squeeze(),
        key=key.squeeze(),
        value=value.squeeze(),
        # attn_mask=mask.squeeze(),
    )
    print(torch_x.size())
    print(torch_scores.size())

    mha = MultiHeadAttention(head_num=head_num, d_model=d_model)
    x, scores = mha(query, key, value, mask)
    print(x.size())
    print(scores.size())
