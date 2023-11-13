#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  version_name=no_env_atten \
  model.feature_interact.pooling=max \
  model.encoder.use_env=False \
  model.encoder.drv_fi=attention \