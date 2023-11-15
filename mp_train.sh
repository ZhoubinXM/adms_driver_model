#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python src/train.py  -m\
  version_name=driver_encoder_self_attention_first_tune \
  model.feature_interact.pooling=first \
  model.encoder.drv_fi=dot,attention \
  model.encoder.drv_embedding_dim=16 \
  model.feature_interact.attention=self \
  data.num_workers=16 \
  model.step_size=20,30 \
  model.lr_decay=0.1 \
