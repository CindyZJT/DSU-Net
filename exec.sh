#!/bin/sh

python train.py --config ./resources/train_config/train_DSGlobal_ce.yaml

python train.py --config ./resources/train_config/train_3dunet_ce.yaml
