# Introduction

This is a PyTorch implementation of ' Multi-Level Alignment Network for Cross-Domain Ship Detection'. This implementation is built on ‘[Domain Adaptive Faster R-CNN for Object Detection in the Wild](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch)' and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
If you find this repository useful, please cite the [original paper](https://www.mdpi.com/2072-4292/14/10/2389). 
# Sturcture
The structure of code is shown below:
1. **configs**  
|- da_faster_rcnn

&emsp;&emsp;&ensp;&thinsp;|- e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml		&emsp;&emsp;&emsp;---The Configuration File of the Dataset DIOR->HRSID

&emsp;&emsp;&ensp;&thinsp;|- e2e_da_faster_rcnn_R_50_C4_hrrsd_to_ssdd.yaml		&emsp;&emsp;&emsp;---The Configuration File of the Dataset HRRSD->SSDD

&emsp;&emsp; …

&emsp;…

2. **maskrcnn_benchmark**		---The Main Implementation File

&emsp;&emsp;&ensp;&thinsp;|- data			&emsp;&emsp;&emsp;&emsp;--- Dataset Reading Code

&emsp;&emsp;&ensp;&thinsp;|- modeling		&emsp;&emsp;---Detector Code

&emsp;&emsp;&ensp;&thinsp;|- backbone

&emsp;&emsp;&ensp;&thinsp;|- cyclegan 		&emsp;&emsp;---Image Layer Alignment

&emsp;&emsp;&ensp;&thinsp;|- da_heads 		&emsp;&emsp;---Convolutional and Instance Layer Alignment

&emsp;&emsp;&ensp;&thinsp;|- detector 

&emsp;&emsp; …

&emsp;…

3. **tools**
  
&emsp;&emsp;&ensp;&thinsp;|- train_net.py
  
&emsp;&emsp;&ensp;&thinsp;|- test_net.py

&emsp;&emsp;…

&emsp;…
# Operating Instructions
1. **training commands**: CUDA_VISIBLE_DEVICES=_device_ python tools/train_net.py --config-file _configuration file_

&emsp;&emsp;For example: CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml"

2. **test commands**: CUDA_VISIBLE_DEVICES=_device_ python tools/test_net.py --config-file _configuration file_ MODEL.WEIGHT _model for test_

&emsp;&emsp;For example: CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_dior_to_hrsid.yaml" MODEL.WEIGHT model/end2end/dior_hrsid_1/model_0120000.pth

# Dataset
The datasets used in this implementation include DIOR, HRRSD, HRSID, HRSID_Inshore, and SSDD. Use the python files in maskrcnn_benchmark/data/
datasets/ to transform the datasets and put them in the **datasets** folder.
