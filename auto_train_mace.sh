#!/bin/bash

# ==================== 配置区域 ====================
MIN_MEMORY=12000           # 需要的显存(MB)，建议设为batch_size=8时实际占用的1.5倍
MAX_WAIT=86400             # 最大等待时间(秒)
SLEEP_INTERVAL=30          # 检查间隔(秒)
LOG_DIR="logs"             # 日志目录
CHECKPOINT_DIR="checkpoints" # 检查点目录

# ==================== 函数定义 ====================
# 获取最佳GPU（显存最大且可用的卡）
get_best_gpu() {
    nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits | \
    awk -F',' '$3 < 1000 {print $1,$2}' | \  # 排除显存占用>1GB的卡
    sort -k2 -rn | head -1 | awk '{print $1}'
}

# 清理GPU残留进程
clean_gpu() {
    echo "[$(date '+%F %T')] 清理GPU$1残留进程..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 | \
    xargs -r kill -9 2>/dev/null
    sleep 3
}

# ==================== 主程序 ====================
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "====== MACE训练启动 ======" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%F %T')" | tee -a "$LOG_FILE"
echo "最低需求显存: ${MIN_MEMORY}MB" | tee -a "$LOG_FILE"

# 等待可用GPU
while true; do
    GPU_ID=$(get_best_gpu)
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID | awk '{print $1}')
    
    if [[ -n "$GPU_ID" && "$FREE_MEM" -ge "$MIN_MEMORY" ]]; then
        echo "[$(date '+%F %T')] ✅ 选中GPU$GPU_ID (可用显存: ${FREE_MEM}MB)" | tee -a "$LOG_FILE"
        break
    fi
    
    echo "[$(date '+%F %T')] ⌛ 等待可用GPU (当前最佳GPU${GPU_ID:-无} 显存: ${FREE_MEM:-0}MB)" | tee -a "$LOG_FILE"
    
    if [ "$SECONDS" -gt "$MAX_WAIT" ]; then
        echo "[$(date '+%F %T')] ❌ 超时未找到可用GPU" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    sleep "$SLEEP_INTERVAL"
done

# 设置GPU并清理残留
export CUDA_VISIBLE_DEVICES=$GPU_ID
clean_gpu $GPU_ID

# 启动训练任务
echo "[$(date '+%F %T')] 🚀 启动训练 on GPU$GPU_ID" | tee -a "$LOG_FILE"
echo "使用的GPU设备: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

mace_run_train \
    --name "llto_simple_model" \
    --model "MACE" \
    --train_file "llto_conservative-train.extxyz" \
    --valid_file "llto_conservative-valid.extxyz" \
    --test_file "llto_conservative-test.extxyz" \
    --E0s "average" \
    --loss "universal" \
    --energy_weight 10 \
    --forces_weight 500 \
    --energy_key "REF_energy" \
    --forces_key "REF_forces" \
    --eval_interval 1 \
    --error_table "PerAtomMAE" \
    --interaction_first "RealAgnosticDensityInteractionBlock" \
    --interaction "RealAgnosticDensityResidualInteractionBlock" \
    --num_interactions 4 \
    --correlation 3 \
    --max_ell 3 \
    --r_max 5.0 \
    --max_L 2 \
    --num_channels 256 \
    --num_radial_basis 8 \
    --MLP_irreps "16x0e" \
    --scaling "rms_forces_scaling" \
    --lr 0.001 \
    --weight_decay 1e-8 \
    --ema \
    --ema_decay 0.99 \
    --scheduler_patience 5 \
    --batch_size 8 \
    --valid_batch_size 8 \
    --pair_repulsion \
    --distance_transform "Agnesi" \
    --max_num_epochs 1600 \
    --patience 200 \
    --amsgrad \
    --device "cuda" \
    --clip_grad 10 \
    --restart_latest \
    --default_dtype "float64" \
    --seed 42 >> "$LOG_FILE" 2>&1

# 训练后处理
echo "[$(date '+%F %T')] 🎉 训练完成! 最终GPU状态:" | tee -a "$LOG_FILE"
nvidia-smi -i $GPU_ID >> "$LOG_FILE"