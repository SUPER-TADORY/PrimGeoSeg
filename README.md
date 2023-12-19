# PrimGeoSeg for 3D Medical Image Segmentation
This repository contains implementation for the generation of PrimGeoSeg dataset (/AVS-DB), pre-training dataset for PrimGeoSeg (/AVS-DB), PrimGeoSeg pre-trained model. 

"Primitive Geometry Segment Pre-training for 3D Medical Image Segmentation", BMVC2023 (Oral) [[Paper](https://papers.bmvc2023.org/0152.pdf)] [[Supplementary](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0152_supp.zip)] [[Video](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0152_video.mp4)]

"Pre-Training Auto-Generated Volumetric Shapes for 3D Medical Image Segmentation", CVPRW2023 (Short Paper) [[Paper](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Tadokoro_Pre-Training_Auto-Generated_Volumetric_Shapes_for_3D_Medical_Image_Segmentation_CVPRW_2023_paper.pdf)]

## Generation for PrimGeoSeg Dataset
Preparing...

## Pre-trained Weights for PrimGeoSeg
|Architecture |Pre-training Size                          |Weights                         |
|----------------|-------------------------------|-----------------------------|
|UNETR|50,000|[UNETR_50000.pt](https://drive.google.com/file/d/1NP_WmRswaOSywqrHw_yTAaFjPhwgi34Y/view?usp=drive_link)|
|SwinUNETR          |5,000|[SwinUNETR_5000.pt](https://drive.google.com/file/d/1NbQqa2jolWbFUYriugNoW4d_vgFykn6m/view?usp=drive_link)|

To load the above chectpoint, please use this code

```
from  monai.networks.nets  import  SwinUNETR, UNETR

# Hyperparameters
roi_x, roi_y, roi_z  =  96, 96, 96
in_channels  =  1
out_channels  =  2  # Set for your target task
feature_size  =  48

# For SwinUNETR
model  =  SwinUNETR(
  img_size=(roi_x, roi_y, roi_z),
  in_channels=in_channels,
  out_channels=out_channels,
  feature_size=feature_size,
  drop_rate=0.0,
  attn_drop_rate=0.0
)

# For UNETR
model  =  UNETR(
  img_size=(roi_x, roi_y, roi_z),
  in_channels=in_channels,
  out_channels=out_channels,
  feature_size=16,
  hidden_size=768,
  mlp_dim=3072,
  num_heads=12,
  pos_embed='conv',
  norm_name='instance',
  conv_block=True,
  res_block=True,
  dropout_rate=0.0
)
	
# Load weight 
state_dict = torch.load('Path to checkpoint path')
model.load_state_dict(state_dict, strict=False)
```


## Pre-training Dataset for PrimGeoSeg
preparing...
|Dataset Size |Pre-training Size                          |
|----------------|-------------------------------|
|50,000|[Google Drive]            |
|5,000          |[Google Drive]            |


## Pre-training & Finetuning
Our experimental code is heavily based on the previous great work []. Pre-training task for PrimGeoSeg dataset is segmentation task which is same as downstream task, and you can perform PrimGeoSeg pre-training by replacing input 3D medical image segmentation dataset to PrimGeoSeg dataset. Also, you can pre-train your original architecture by replacing the architecture part in the code. 

## Cite

 ```
@inproceedings{tadokoro2023primgeoseg,
  title={Primitive Geometry Segment Pre-training for 3D Medical Image Segmentation},
  author={Ryu Tadokoro, Ryosuke Yamada, Kodai Nakashima, Ryo Nakamura, Hirokatsu Kataoka},
  booktitle={The 34th British Machine Vision Conference Proceedings},
  year={2023}
}
@inproceedings{tadokoro2023avsdb,
  title={Pre-Training Auto-Generated Volumetric Shapes for 3D Medical Image Segmentation},
  author={Ryu Tadokoro, Ryosuke Yamada, Hirokatsu Kataoka},
  booktitle={IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2023}
}
```
