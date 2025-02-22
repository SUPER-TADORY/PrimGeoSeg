# PrimGeoSeg for 3D Medical Image Segmentation
![Main Image](figure/main.png)

This repository contains implementation for the generation of PrimGeoSeg dataset (/AVS-DB), pre-training dataset for PrimGeoSeg (/AVS-DB), PrimGeoSeg pre-trained model. 

"Primitive Geometry Segment Pre-training for 3D Medical Image Segmentation", BMVC2023 (Oral) [[Proceedings](https://proceedings.bmvc2023.org/152/)] [[Paper](https://papers.bmvc2023.org/0152.pdf)] [[Supplementary](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0152_supp.zip)] [[Video](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0152_video.mp4)]

"Pre-Training Auto-Generated Volumetric Shapes for 3D Medical Image Segmentation", CVPRW2023 (Short Paper)

## Generation for PrimGeoSeg Dataset
![Data Construction](figure/data_generation.png)
To construct the pre-training dataset for PrimGeoSeg, please run the following code. You can customize the dataset by modifying the hyperparameters.
```
cd data_generation
pip install -r requirements.txt
bash run.sh
```

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

|Dataset Size |Pre-training Size                          |
|----------------|-------------------------------|
|50,000|[Google Drive]            |
|5,000          |[Google Drive]            |


## Pre-training & Finetuning
This project is based on the work found in [the MONAI Research Contributions](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR), specifically the SwinUNETR/BTCV implementation. Our approach applies this framework for both pre-training & finetuning.

### Pre-Training with PrimGeoSeg
The key contribution of our work is using the PrimGeoSeg dataset for pre-training in a 3D medical image segmentation context. Although [the MONAI codebase](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) was originally developed for datasets like BTCV, it can be easily adapted by simply substituting PrimGeoSeg as the input dataset. This allows you to leverage the same training framework for pre-training, ensuring seamless integration with the downstream segmentation tasks of interest. To achieve optimal results, we recommend aligning the hyperparameters with those provided in the main text and supplementary materials of our paper.

###  Customizing the Architecture
In addition to dataset adaptation, our project offers flexibility in terms of architectural modifications. If you're looking to implement your own architecture, this can be easily achieved by replacing the architecture component in the codebase with your design. This feature allows for extensive experimentation and adaptation, tailoring the project to meet various research needs.

## Erratum
[2024/07] As mentioned in the supplementary materials of the BMVC paper, the MSD results in the CVPRW paper are reported as the average Dice score of both the target and background classes. In contrast, the BMVC paper records the Dice score for the target class only. Additionally, the BMVC paper uses hyperparameters that are more closely aligned with previous research for MSD experiments.


## Cite

 ```
@inproceedings{Tadokoro_2023_BMVC,
	author={Ryu Tadokoro and Ryosuke Yamada and Kodai Nakashima and Ryo Nakamura and Hirokatsu Kataoka},
	title={Primitive Geometry Segment Pre-training for 3D Medical Image Segmentation},
	booktitle={34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
	publisher={BMVA},
	year={2023},
	url={https://papers.bmvc2023.org/0152.pdf}
}
```
