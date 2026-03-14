export CUDA_VISIBLE_DEVICES=6,7
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=lo
# NCCL_DEBUG=INFO

export EXP_NAME="Dome-L-VisDrone"

torchrun --master_port=7789 --nproc_per_node=2 train.py \
     -c configs/dome/Dome-L-VisDrone.yml --seed=0  2>&1 | tee "logs/${EXP_NAME}-$(date +%Y%m%d_%H%M%S).log"