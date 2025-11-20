export CUDA_VISIBLE_DEVICES=0,1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=lo
# NCCL_DEBUG=INFO
torchrun --master_port=7788 --nproc_per_node=2 train.py \
     -c configs/dome/Dome-L-AITOD.yml --seed=0