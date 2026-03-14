export CUDA_VISIBLE_DEVICES=2,3
# export NCCL_SOCKET_IFNAME=lo
# export SAVE_TEST_VISUALIZE_RESULT=False
# export SAVE_INTERMEDIATE_VISUALIZE_RESULT=True
torchrun --master_port=7778 --nproc_per_node=2 train.py -c configs/dome/Dome-M-VisDrone.yml --test-only -r logs/dome-m-visdrone_converted.pth