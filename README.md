# image semantic segmentation package

A image semantic segmentation toolbox (single GPU) contains some common semantic segmentation algorithms. The code is implemented by Pytorch.

### Requires

  1. pytorch >= 1.0.0
  2. python >= 3.6
  
### Usage

### Performance



#### PSPNet(ResNet50)
| ADE20K        |    pixAcc    |    mIoU    |
| --------   | -------:  | :------:  |
| paper    |    80.04   |   41.68  |
| my code(without background)  |    77.10     |    39.0   |
| my code(with background)  |     \$12     |     12     |

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




