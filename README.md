## image semantic segmentation package

A image semantic segmentation toolbox (single GPU) contains some common semantic segmentation algorithms. The code is implemented by Pytorch.

### Requires

  1. pytorch >= 1.0.0
  2. python >= 3.6
  
### Usage

```
 python train.py --option1 value1 --option2 value2 ...
```

### Performance

#### PSPNet(ResNet50)

| ADE20K    |   pixAcc    |    mIoU    |
| -------- | -------:  | :------: |
| [paper](https://github.com/hszhao/PSPNet) [1]  |    80.04   |   41.68  |
| my code(with background)  |    70.4   |   32.7   |

##### Discussion and details:
```
 epoch: 30
 learning rate scheduler: poly
 batch size: 12
 image size: 384
 nbr_classes: 150(without background, standard ade20k) / 151 (with background)
```
  In the original paper, authors run their experiments on the standard ADE20k(150 classes, without background). 
  But I regard the background (i.e. labeled 0 in the original mask) as a category and the output dimensionality of the PSPNet is 151 in my code.
  Therefore, the performance gap mainly comes from three aspects：
  1) I add the background class to the dataset, which may lead to category imbalance problems and increases the complexity of the model.
  2) Due to limited video memory on a single GPU, I set the batch_size to 12 and image_size to 384 instead of 16 and 473 in the original paper. 
  3) In addition, the experiments in the original paper used multiple GPUs, which means a larger batch_size can be set to make Synchronization Batch Normalization layers more effective.
    
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


### References
[1] [Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1612.01105)

