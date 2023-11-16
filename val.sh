#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  ckpt_path=data/checkpoint/epoch_051_0.270705.ckpt \
  model.encoder.drv_embedding_dim=16 \
  model.encoder.drv_fi=attention \
  model.feature_interact.pooling=first \
  model.feature_interact.attention=self \
  model.feature_interact.plot=false \
  version_name=driver_encoder_16_self_atten_first \

