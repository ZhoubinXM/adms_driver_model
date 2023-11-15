#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python src/train.py \
  version_name=driver_encoder_self_attention_first \
  model.feature_interact.pooling=first \
  model.encoder.drv_fi=dot \
  model.encoder.drv_embedding_dim=16 \
  model.feature_interact.attention=self \
  data.num_workers=16 \
