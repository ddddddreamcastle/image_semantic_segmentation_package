# image semantic segmentation package

A image semantic segmentation toolbox (single GPU) contains some common semantic segmentation algorithms. The code is implemented by Pytorch.

### Requires

  1. pytorch >= 1.0.0
  2. python >= 3.6
  
### Usage

### Performance



#### PSPNet(ResNet50)
| ADE20K    |   pixAcc    |    mIoU    |
| -------- | -------:  | :------:  |
| paper  |    80.04   |   41.68  |
| my code(without background)  |   77.10   |  39.0  |
| my code(with background)  |    \$12   |   12   |
##### Discussion and details:
`epoch: 30

 earning rate scheduler: poly
 
 batch size: 12
 
 image size: 384
 
 nbr_classes: 150(without background, standard ade20k) / 151 (with background)
 
`
In the original paper, authors run their experiments on the standard ADE20k(150 classes, without background). 
But I regard the background (i.e. labeled 0 in the original mask) as a class and the output dimensionality of the PSPNet is 151 in my code.
Therefore, the performance gap mainly comes from two aspects：
1) I add the background class to the dataset, which is possible to cause class imbalance problems and increases the complexity of the model.
2) Due to limited video memory on a single GPU, I set the batch size to 12 and image size to 384 instead of 16 and 473 in the original paper. In addition, the experiments in the original paper used multiple GPUs.
    
### TODO

- [x] PSPNet
- [ ] Deeplabv3
- [ ] RefineNet
- [ ] FPN
- [ ] LinkNet
- [ ] SegNet
- [ ] FCN
- [ ] Unet
- [ ] Unet++




