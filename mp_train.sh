#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python src/train.py  -m\
  version_name=driver_encoder_self_atten_embed_dim_fi_cross_attention \
  model.feature_interact.pooling=max,mean \
  model.encoder.drv_fi=attention,dot \
  model.encoder.drv_embedding_dim=16 \
  model.feature_interact.attention=cross \