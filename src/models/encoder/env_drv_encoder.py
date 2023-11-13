import torch
from torch import nn

from src.models.layers import *
from src.models.encoder.encoder import *


class EnvDrvEncoder(Encoder):
    """Env and Driver modeling"""

    def __init__(
            self,
            # env
            use_env: bool = True,
            proposal_num: int = 6,
            target_in_dim: int = 9,
            agent_in_dim: int = 9,
            encoder_out_dim: int = 256,
            agent_steps: int = 49,
            # map
            use_map: bool = True,
            map_in_dim: int = 6,
            map_steps: int = 19,
            # drv
            use_drv: bool = True,
            num_dense_feature: int = 27,
            num_sparse_features=[13, 3, 6, 2, 5, 5, 18, 4, 2, 2, 3, 16],
            drv_embedding_dim=256,
            mlp_dims=[256, 128, 64],
            drv_fi: str = "dot",
            dropout: float = 0.1,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_env = use_env
        self.use_map = use_map
        self.use_drv = use_drv

        self.num_sparse_features = num_sparse_features
        embedding_dims = [
            drv_embedding_dim for _ in range(len(self.num_sparse_features))
        ]
        self.mlp_dims = mlp_dims
        self.drv_fi = drv_fi

        if use_env:
            self.use_map = True
            self.queries = nn.Parameter(torch.eye(proposal_num,
                                                  encoder_out_dim),
                                        requires_grad=True)
            self.target_encoder = TargetEncoder(target_in_dim, encoder_out_dim)
            self.target_vec_encoder = TargetVecEncoder(target_in_dim,
                                                       encoder_out_dim // 2,
                                                       agent_steps, 3)
            self.agent_vec_encoder = AgentVecEncoder(agent_in_dim,
                                                     encoder_out_dim // 2,
                                                     agent_steps, 3)
            self.cross_attention = CrossAttentionLayer(
                head_num=4,
                d_model=encoder_out_dim,
                d_ff=4 * encoder_out_dim,
                norm_dim=None,
                dropout=dropout,
            )
        self.map_encoder = (MapEncoder(map_in_dim, encoder_out_dim //
                                       2, map_steps, 3) if use_map else None)
        if use_drv:
            if self.drv_fi == 'mlp':
                pass
            else:
                self.driver_encoder = DriverModel(
                    num_dense_features=num_dense_feature,
                    num_sparse_features=self.num_sparse_features,
                    embedding_dims=embedding_dims,
                    mlp_dims=self.mlp_dims,
                    interaction_type=self.drv_fi,
            )
        
        self.apply(weight_init)
    
    def _forward(
        self,
        target: torch.Tensor,
        agent: torch.Tensor,
        agent_mask: torch.Tensor,
        lane: Optional[torch.Tensor],
        lane_mask: Optional[torch.Tensor],
        sparse_feature: Optional[torch.Tensor],
        dense_feature: Optional[torch.Tensor],
        target_pos_in_ego: Optional[torch.Tensor],
        img: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """abstract method implemented

        Args:
            target (torch.Tensor): _description_
            agent (torch.Tensor): _description_
            agent_mask (torch.Tensor): _description_
            lane (Optional[torch.Tensor]): _description_
            lane_mask (Optional[torch.Tensor]): _description_
            sparse_feature (Optional[torch.Tensor]): _description_
            dense_feature (Optional[torch.Tensor]): _description_
            target_pos_in_ego (Optional[torch.Tensor]): _description_
            img (Optional[torch.Tensor]): _description_

        Returns:
            torch.Tensor: acsr [B, 106, 1, 256]
        """
        acsr, mask, driver_enc = None, None, None
        if self.use_env:
            proposal_queries = self.queries.view(1, 1,
                                                 *self.queries.shape).repeat(
                                                     *target.shape[:2], 1, 1)

            agent_mask_n, agent_mask_c, agent_mask_h, agent_mask_w = agent_mask.shape

            target_enc = self.target_encoder(target)
            target_vec_enc = self.target_vec_encoder(target)
            agent_enc = self.agent_vec_encoder(
                agent,
                agent_mask[0:agent_mask_n, 1:agent_mask_c, 0:agent_mask_h,
                           0:agent_mask_w],
            )
            if self.use_map:
                lane_enc = self.map_encoder(lane, lane_mask)

            # cross attention for proposals
            proposals, _ = self.cross_attention(proposal_queries.transpose(
                1, 2),
                                                target_enc.transpose(1, 2),
                                                mask=None)
            # self attention for anchors
            acsr = torch.cat((target_vec_enc, agent_enc), dim=1)

            mask = agent_mask[0:agent_mask_n, 0:agent_mask_c, 0:1,
                              0:agent_mask_w]
            if self.use_map:
                acsr = torch.cat((acsr, lane_enc), dim=1)
                lane_mask_n, lane_mask_c, lane_mask_h, lane_mask_w = lane_mask.shape
                mask = torch.cat(
                    (mask, lane_mask[0:lane_mask_n, 0:lane_mask_c, 0:1,
                                     0:lane_mask_w]),
                    dim=1,
                )
            mask = mask.transpose(1, -1)

        if self.use_drv:
            driver_enc = self.driver_encoder(dense_feature, sparse_feature)

        return acsr, mask, driver_enc
