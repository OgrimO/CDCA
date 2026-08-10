# 默认设备ID为0
DEVICE_ID=0

# 解析命令行参数
for arg in "$@"; do
    if [[ $arg == --device_id=* ]]; then
        DEVICE_ID="${arg#*=}"
    fi
done


#!/bin/bash

# CDCA
# layer = 2
python CDCA.py --dataset=Cora --vs_number=40 --loss_2hop_weight=1 --loss_2hop_suppress_weight=2  --neighbor_suppress_hops=1 --selection_method=target_cluster_distance
# layer = 3
python CDCA.py --dataset=Cora --vs_number=40 --loss_2hop_weight=2 --loss_2hop_suppress_weight=2  --dis_weight=1 --neighbor_suppress_hops=1 --selection_method=target_cluster_distance --layer=3
# layer = 4
python CDCA.py --dataset=Cora --vs_number=40 --loss_2hop_weight=2 --loss_2hop_suppress_weight=2  --dis_weight=1 --neighbor_suppress_hops=2 --selection_method=target_cluster_distance --layer=4

# layer = 2
python CDCA.py --dataset=Citeseer --vs_number=40 --loss_target_weight=1 --loss_2hop_weight=2 --loss_2hop_suppress_weight=1 --homo_loss_weight=0 --neighbor_suppress_hops=1 --selection_method=target_cluster_distance
# layer = 3
python CDCA.py --dataset=Citeseer --vs_number=40 --loss_target_weight=2 --loss_2hop_weight=2 --loss_2hop_suppress_weight=6 --dis_weight=0.5 --homo_loss_weight=0 --neighbor_suppress_hops=2 --selection_method=target_cluster_distance --layer=3
# layer = 4
python CDCA.py --dataset=Citeseer --vs_number=40 --loss_target_weight=2 --loss_2hop_weight=2 --loss_2hop_suppress_weight=6 --dis_weight=0.5 --homo_loss_weight=0 --neighbor_suppress_hops=2 --selection_method=target_cluster_distance --layer=4

# layer = 2
python CDCA.py --dataset=Pubmed --vs_number=90 --epochs=400 --loss_2hop_weight=1 --loss_2hop_suppress_weight=2 --homo_loss_weight=0 --neighbor_suppress_hops=1 --selection_method=target_cluster_distance
# layer = 3
python CDCA.py --dataset=Pubmed --vs_number=90 --epochs=400 --loss_2hop_weight=1 --loss_2hop_suppress_weight=6 --homo_loss_weight=0 --neighbor_suppress_hops=1 --selection_method=target_cluster_distance --layer=3
# layer = 4
python CDCA.py --dataset=Pubmed --vs_number=90 --epochs=400 --loss_2hop_weight=2 --loss_2hop_suppress_weight=2 --homo_loss_weight=0 --neighbor_suppress_hops=2 --selection_method=target_cluster_distance --layer=4


# layer = 2
python CDCA.py --dataset=ogbn-arxiv --vs_number=565 --trojan_epochs=400 --hidden=128 --loss_target_weight=1 --loss_2hop_suppress_weight=1 --loss_2hop_weight=1 --neighbor_suppress_hops=1 --homo_loss_weight=0 --selection_method=target_cluster_distance
# layer = 3
python CDCA.py --dataset=ogbn-arxiv --vs_number=565 --trojan_epochs=400 --hidden=128 --loss_target_weight=1 --loss_2hop_suppress_weight=1 --loss_2hop_weight=2 --neighbor_suppress_hops=1 --homo_loss_weight=0 --selection_method=target_cluster_distance --layer=3
# layer = 4
python CDCA.py --dataset=ogbn-arxiv --vs_number=565 --trojan_epochs=400 --hidden=128 --loss_target_weight=1 --loss_2hop_suppress_weight=1 --loss_2hop_weight=2 --neighbor_suppress_hops=1 --homo_loss_weight=0 --selection_method=target_cluster_distance --layer=4

@echo off
setlocal enabledelayedexpansion
