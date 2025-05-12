export CUDA_VISIBLE_DEVICES=0
export SAVE_INTERMEDIATE_VISUALIZE_RESULT=True
python tools/inference/torch_inf.py --config configs/dome/custom/DOME-AITODv2-800_800-original-3e4d-light-ccm.yml \
       --resume output/dfine_hgnetv2_m_aitod/best_stg1.pth \
       --input /data/lihb/Datasets/aitod/aitod/images/test/3ce09a9bd.png \
       --annotation /data/lihb/Datasets/aitod/aitod/annotations/aitodv2_test.json \
       --device cuda:0