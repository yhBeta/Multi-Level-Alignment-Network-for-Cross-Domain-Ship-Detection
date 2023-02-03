
CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml" MODEL.WEIGHT model/end2end/ablations_studies/cycgan_feat_1/model_0005000.pth

