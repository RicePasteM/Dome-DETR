export CUDA_VISIBLE_DEVICES=4,5,6,7
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=lo
# NCCL_DEBUG=INFO

export EXP_NAME="Dome-M-AITOD"

torchrun --master_port=7788 --nproc_per_node=4 train.py \
     -c configs/dome/Dome-M-AITOD.yml --seed=0  2>&1 | tee "logs/${EXP_NAME}-$(date +%Y%m%d_%H%M%S).log"