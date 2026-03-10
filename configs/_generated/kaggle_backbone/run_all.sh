#!/usr/bin/env bash
set -e

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_eva02
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_eva02 --experiment_name backbone_eva02

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_megadescriptor_l
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_megadescriptor_l --experiment_name backbone_megadescriptor_l

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_miew_id
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_miew_id --experiment_name backbone_miew_id

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dino_small
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dino_small --experiment_name backbone_dino_small

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov2_base
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov2_base --experiment_name backbone_dinov2_base

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov3_base
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov3_base --experiment_name backbone_dinov3_base

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov3_large
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov3_large --experiment_name backbone_dinov3_large

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov2_for_wildlife
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_dinov2_for_wildlife --experiment_name backbone_dinov2_for_wildlife

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_convnext_v2
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_convnext_v2 --experiment_name backbone_convnext_v2

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_efficientnet_b4
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_efficientnet_b4 --experiment_name backbone_efficientnet_b4

python src/jaguar/experiments/experiment_setup.py --setup_name kaggle_base --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_swin_transformer
python src/jaguar/main.py --base_config base/kaggle_base --experiment_config _generated/kaggle_backbone/backbone_swin_transformer --experiment_name backbone_swin_transformer

