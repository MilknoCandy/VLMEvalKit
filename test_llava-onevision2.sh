#!/bin/bash

echo "=========================================================="
echo "开始使用 LLaVA-OneVision-2-8B-Instruct 模型进行评估测试"
echo "请确保您已配置好对应的环境并正确挂载了模型权重"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_NCCL_TRACE_BUFFER_SIZE=4194304
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_IDS[@]}

export LMUData=/nfs/SDW/datasets/LMUData
echo "LMUData path: ${LMUData}"

export HF_HOME=/nfs/huggingface_cache
echo "HF_HOME path: ${HF_HOME}"

MODEL_NAME=LLaVA-OneVision-2-8B-Instruct

MASTER_PORT=19507

TASK=VSI-Bench_1fps


# 限制可见的 GPU 卡（您可以根据实际设备数量调整，这里默认使用 4 张卡）
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 这里使用 VLMEvalKit 原生支持的 VSI-Bench 数据源进行测试评估
ACCELERATE_CPU_AFFINITY=1 torchrun --master_port=$MASTER_PORT \
  --nproc-per-node=8 run.py \
  --model ${MODEL_NAME} \
  --data ${TASK} \
  --mode all \
  --judge exact_matching \
  --work-dir outputs/llava-onevision2/${TASK}

echo "=========================================================="
echo "评估完成！结果输出在 outputs/llava-onevision2/${TASK} 目录下"
echo "=========================================================="
