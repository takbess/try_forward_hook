#!/bin/bash

# Train all models
# python train.py
python train_teacher.py
python train_kd.py
python train_crosskd.py

# Visualize log
python plot_log.py   --log_path log/resnet18.log   --out_prefix log_plot/resnet18
python plot_log.py   --log_path log/resnet34.log   --out_prefix log_plot/resnet34
python plot_log.py   --log_path log/kd.log         --out_prefix log_plot/kd
python plot_log.py   --log_path log/crosskd.log    --out_prefix log_plot/crosskd

# Visualize feature stats log
python visualize_feat_log_all.py

