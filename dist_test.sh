export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_SOCKET_IFNAME=lo
# export SAVE_TEST_VISUALIZE_RESULT=False
# export SAVE_INTERMEDIATE_VISUALIZE_RESULT=True
torchrun --master_port=7778 --nproc_per_node=4 train.py -c configs/dome/Dome-M-AITOD.yml --test-only -r output/dome_m_aitod/best_stg2_converted.pth