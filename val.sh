#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  ckpt_path=data/checkpoint/epoch_056_1.433609.ckpt \
  model.encoder.drv_embedding_dim=16 \
  model.encoder.drv_fi=attention \
  model.feature_interact.pooling=first \
  model.feature_interact.attention=self \
  model.feature_interact.plot=false \
  data.train_file=train_val_quantile.pkl \
  data.test_file=test_quantile.pkl \
  model.lbl_proc=none \
  model.mode=equal_width_cls \
  model.decoder.output_dim=5 \
  version_name=equal_width_cls \

