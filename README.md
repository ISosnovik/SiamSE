# Scale Equivariance Improves Siamese Tracking
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scale-equivariance-improves-siamese-tracking/visual-object-tracking-on-otb-2013)](https://paperswithcode.com/sota/visual-object-tracking-on-otb-2013?p=scale-equivariance-improves-siamese-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scale-equivariance-improves-siamese-tracking/visual-object-tracking-on-otb-2015)](https://paperswithcode.com/sota/visual-object-tracking-on-otb-2015?p=scale-equivariance-improves-siamese-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scale-equivariance-improves-siamese-tracking/visual-object-tracking-on-vot2016)](https://paperswithcode.com/sota/visual-object-tracking-on-vot2016?p=scale-equivariance-improves-siamese-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scale-equivariance-improves-siamese-tracking/visual-object-tracking-on-vot2017)](https://paperswithcode.com/sota/visual-object-tracking-on-vot2017?p=scale-equivariance-improves-siamese-tracking)

Ivan Sosnovik*, Artem Moskalev*, and Arnold Smeulders [Scale Equivariance Improves Siamese Tracking](https://arxiv.org/abs/2007.09115), WACV 2021.


## Introduction

*Siamese trackers turn tracking into similarity estimation between a template and the candidate regions in the frame. Mathematically, one of the key ingredients of success of the similarity function is translation equivariance. Non-translation-equivariant architectures induce a positional bias during training, which hinders accurate localization. In real-life scenarios, however, the target undergos more transformations than just translation, e.g. rotation and scaling. In this work we focus on the later and demonstrate that extending Siamese tracking with built-in scale-equivariance improves tracking quality.*

<br>

<div align="center">
  <img src="src/pallete.gif" , width="100%"/>
  <!-- <p>Example SiamFC, SESiamFC.</p> -->
</div>

<br>
<br>

## Results

<br>

| Models  | OTB-2013 | OTB-2015 | VOT2016 | VOT2017 |
| :------ | :------: | :------: | :------: | :------: |
| SiamFC+  | 0.67 | 0.64 | 0.30 | 0.23 |
| SE-SiamFC  | 0.68 | 0.66 | 0.36 | 0.27 |

Raw results and models are available [here](https://drive.google.com/drive/folders/1QnVId75-U2AWVcyDdDz7Qj_6Ob2DPIRk?usp=sharing)

**Environment:**
The code is developed with Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz GPU: NVIDIA Titan RTX, 

## Requirements
To train models and to run evaluations, you need the following dependencies
```
yacs
scipy
shapely
opencv-python
numpy
pytorch
torchvision
pyyaml
```

You will need to install python API for MATLAB engine to run the VOT benchmarks.

## Quick start
1. Download the weights for model initialization from [here](https://drive.google.com/drive/folders/1QnVId75-U2AWVcyDdDz7Qj_6Ob2DPIRk?usp=sharing). We used `SESiamFCResNet22_pretrained.pth`. You can also run the script `transfer_weights.py` to generate your own weights.
2. Download the preprocessed training and testing datasets. Use the instructions provided 
[here](https://github.com/researchmm/SiamDW/blob/master/lib/tutorials/train.md)
3. Adjust the training config in `configs/train.yaml` according to your tasks.
4. Run the training script
```bash
CUDA_VISIBLE_DEVICES=0 python train_siamfc.py --cfg configs/train.yaml
```
5. To run the tracker from a snapshot, simply do the following
```
CUDA_VISIBLE_DEVICES=0 python test_siamfc.py \
    --checkpoint snapshot/SESiamFCResNet22/checkpoint_otb.pth \
    --dataset OTB2015 \
    --dataset_root ~/datasets/ \
    --cfg configs/test.yaml 

```
6. When the output is generated, you are able to evaluate the performance of the tracker. Use `lib/core/eval_otb.py` or `lib/core/eval_vot.py`. For example,
```
python lib/core/eval_otb.py \
    --dataset OTB2015 \
    --dataset_root ~/datasets \
    --result_path results \
    --tracker_reg "SE*" \

```


## Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.

## Citation
```
If you found this work useful in your research, please consider citing

@InProceedings{Sosnovik_2021_WACV,
    author    = {Sosnovik, Ivan and Moskalev, Artem and Smeulders, Arnold W.M.},
    title     = {Scale Equivariance Improves Siamese Tracking},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2765-2774}
}
```

## License
Licensed under an MIT license.
