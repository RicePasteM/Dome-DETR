export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_SOCKET_IFNAME=lo
# export SAVE_TEST_VISUALIZE_RESULT=False
# export SAVE_INTERMEDIATE_VISUALIZE_RESULT=True
torchrun --master_port=7778 --nproc_per_node=8 train.py -c configs/dome/Dome-L-AITOD.yml --test-only -r checkpoints/Dome-L-AITOD-best.pth