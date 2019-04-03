import argparse
from models.model_manager import Manager

parser = argparse.ArgumentParser()
# model and training
parser.add_argument('--model', type=str, default='pspnet', help='model name')
parser.add_argument('--model_pretrained', type=bool, default=True)
parser.add_argument('--model_pretrain_path', type=str, default='./logs/checkpoints/trainlog_pspnet_resnet50_best.pt',
                    help='If `model_pretrained` is True, this param is necessary, or else will be neglected.')
parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name')
parser.add_argument('--backbone_pretrained', type=str, default='resnet50',
                    help='If `model_pretrained` is True, this param will be neglected.')
parser.add_argument('--backbone_pretrained_path', type=str, default='./weights/resnet50.pth',
                    help='If `model_pretrained` is True, this param will be neglected.')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--mode', type=str, default='pred', help='options include `train` and `pred` ')

# the learning rate setting refer to https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/segmentation/option.py
parser.add_argument('--learning_rate', type=float, default=None,
                    help='learning rate. If None, learning rate will be adjusted adaptively according to the dataset')
parser.add_argument('--lr_scheduler', type=str, default='polynomial')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--supervision', default= True)
parser.add_argument('--supervision_weight', type=float, default=0.4)

#dataset
parser.add_argument('--dataset', type=str, default='ade20k')
parser.add_argument('--data_path', type=str, default='/DATA1000/zhaopeng/workspace/research/dishes_recognition/data/ADEChallengeData2016')
parser.add_argument('--image_size', type=int, default=384)
parser.add_argument('--dataloader_workers', type=int, default=4)
# requires the id of background category is 0
parser.add_argument('--use_background', type=bool, default=True)

# checkpoint and saving model
parser.add_argument('--checkpoint_root_path', type=str, default='./logs/checkpoints/')
parser.add_argument('--checkpoint_prefix_name', type=str, default='trainlog')
parser.add_argument('--load_checkpoint', type=bool, default=False)
parser.add_argument('--load_checkpoint_path', type=bool, default=False)

args = parser.parse_args()
manager = Manager(args)
# manager.fit()
manager.predict('/DATA1000/zhaopeng/workspace/research/dishes_recognition/data/ADEChallengeData2016/val/images/ADE_val_00001099.jpg',
                '/DATA1000/zhaopeng/test/image_semantic_segment_package/output/')


