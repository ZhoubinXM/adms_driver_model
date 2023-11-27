#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python src/train.py \
  version_name=equal_width_cls \
  model.feature_interact.pooling=first \
  model.encoder.drv_fi=attention \
  model.encoder.drv_embedding_dim=16 \
  model.encoder.dropout=0.5 \
  model.feature_interact.attention=self \
  model.feature_interact.dropout=0.5 \
  data.num_workers=16 \
  data.train_file=train_val_quantile.pkl \
  data.test_file=test_quantile.pkl \
  model.lr_decay=0.2 \
  model.step_size=50 \
  model.lbl_proc=none \
  model.mode=equal_width_cls \
  model.decoder.output_dim=5 \
