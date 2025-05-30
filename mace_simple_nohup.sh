#!/bin/bash
# mace_simple_nohup.sh

# 去掉所有 #SBATCH 行，添加nohup适用的设置
echo "开始训练MACE简单模型 - nohup版本"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo "模型参数: num_channels=128, max_L=1, num_interactions=2"

# 创建日志目录
mkdir -p logs checkpoints

export CUDA_VISIBLE_DEVICES=3

# 训练参数
forces_weight=("100")
max_num_epochs=("800")
len=${#forces_weight[@]}

for ((i=0; i<len; i++)); do
    echo "开始训练配置 $((i+1))/$len"
    
    mace_run_train \
        --name="llto_simple_model" \
        --model="MACE" \
        --train_file="llto_conservative-train.extxyz" \
        --valid_file="llto_conservative-valid.extxyz" \
        --test_file="llto_conservative-test.extxyz" \
        --E0s="average" \
        --loss='universal' \
        --energy_weight=1 \
        --forces_weight=${forces_weight[i]} \
        --energy_key='REF_energy' \
        --forces_key='REF_forces' \
        --eval_interval=1 \
        --error_table='PerAtomMAE' \
        --interaction_first="RealAgnosticDensityInteractionBlock" \
        --interaction="RealAgnosticDensityResidualInteractionBlock" \
        --num_interactions=2 \
        --correlation=3 \
        --max_ell=3 \
        --r_max=5.0 \
        --max_L=1 \
        --num_channels=128 \
        --num_radial_basis=8 \
        --MLP_irreps="16x0e" \
        --scaling='rms_forces_scaling' \
        --lr=0.01 \
        --weight_decay=1e-8 \
        --ema \
        --ema_decay=0.99 \
        --scheduler_patience=5 \
        --batch_size=10 \
        --valid_batch_size=10 \
        --pair_repulsion \
        --distance_transform="Agnesi" \
        --max_num_epochs=${max_num_epochs[i]} \
        --patience=50 \
        --amsgrad \
        --device=cuda \
        --seed=42 \
        --clip_grad=10 \
        --restart_latest \
        --default_dtype="float64"
done

