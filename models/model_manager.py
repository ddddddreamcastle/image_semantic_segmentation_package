from models.nn import get_model
from datasets import get_train_val_loader
import torch
from loss import SegmentationLoss
from torch import nn
from utils.learning_rate_scheduler import LearningRateScheduler
from utils.learning_rate_scheduler import lr_parse
from tqdm import tqdm
from utils.meter import SegmentationErrorMeter
from utils.recorder import save_checkpoint
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from utils.color import add_color
class Manager(object):

    def __init__(self, args):
        args = lr_parse(args)
        self.kwargs = vars(args)
        self.epochs = args.epochs
        self.model = get_model(name=args.model, kwargs=self.kwargs)
        self.train_loader, self.val_loader = get_train_val_loader(args.dataset, **self.kwargs)

        parameters = self.model.get_parameters_as_groups(args.learning_rate)

        self.optimizer = torch.optim.SGD(parameters, lr=args.learning_rate,
                                          weight_decay=args.weight_decay, momentum=args.momentum)

        if args.supervision:
            self.criterion = SegmentationLoss((nn.CrossEntropyLoss(), nn.CrossEntropyLoss()),
                                     (1, args.supervision_weight))
        else:
            self.criterion = SegmentationLoss((nn.CrossEntropyLoss(), ),
                                         (1, ))

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.best_performance = 0
        self.start_epoch = 0
        self.lr_scheduler = LearningRateScheduler(args.lr_scheduler, args.learning_rate, args.epochs,
                                                  len(self.train_loader),
                                                  lr_decay_every=10, decay_rate = 0.1)

        if args.load_checkpoint:
            checkpoint = torch.load(args.load_checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.best_performance = checkpoint['best_performance']
            self.start_epoch = checkpoint['start_epoch']
            print("checkpoint loaded successfully")

    def __do_epoch(self, epoch):
        train_loss = 0
        tqdm_bar = tqdm(self.train_loader)
        meter = SegmentationErrorMeter(['pixAcc', 'mIoU'], self.model.nbr_classes)
        for i, (image, target) in enumerate(tqdm_bar):
            cur_lr = self.lr_scheduler.adjust_learning_rate(self.optimizer, i, epoch)
            self.optimizer.zero_grad()
            # target = torch.randint(0,100, (12, 96, 96)).long()
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            preds = self.model(image)
            loss = self.criterion(preds, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # pred = preds
            # if self.model.deep_supervision:
            #     pred = preds[0]
            # meter.add(pred, target)
            # pixAcc, mIoU = meter.values()
            # tqdm_bar.set_description('Lr: {:.4}, Train loss: {:.3}, Train pixAcc: {:.3}, Train mIoU: {:.3}'
            #                          .format(cur_lr, train_loss/(i+1), pixAcc, mIoU))
            tqdm_bar.set_description('Lr: {:.4}, Train loss: {:.3}'
                                     .format(cur_lr, train_loss/(i+1)))

    def __do_validation(self, epoch):
        tqdm_bar = tqdm(self.val_loader)
        meter = SegmentationErrorMeter(['pixAcc', 'mIoU'], self.model.nbr_classes)
        pixAcc, mIoU = 0, 0
        for i, (image, target) in enumerate(tqdm_bar):
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            preds = self.model(image)
            pred = preds
            if self.model.deep_supervision:
                pred = preds[0]
            meter.add(pred, target)
            pixAcc, mIoU = meter.values()
            tqdm_bar.set_description('pixAcc: {:.3}, mIoU: {:.3}'.format(pixAcc, mIoU))
        performance = (pixAcc + mIoU)/2
        is_best = False
        if performance > self.best_performance:
            self.best_performance = performance
            is_best = True
        save_checkpoint(weights=self.model.state_dict(), epoch=epoch,
                        best_performance=self.best_performance, is_best=is_best, **self.kwargs)

    def fit(self):
        for epoch in range(self.epochs):
            print('Epoch : {}'.format(epoch + 1))
            # train step
            self.model.train()
            self.__do_epoch(epoch)

            # val step
            self.model.eval()
            with torch.no_grad():
                self.__do_validation(epoch)

    def predict(self, path, output_path):
        def preprocessing(img):
            ori = img.resize((self.kwargs['image_size'], self.kwargs['image_size']), Image.BILINEAR)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ])
            return ori, transform(ori)

        def color(ori, pred):
            cm = np.argmax(pred, axis=0)
            color_cm = add_color(cm, self.model.nbr_classes)
            img = 0.5 * ((color_cm * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])*255 + \
                   0.5 * np.array(ori)
            return img

        def single_image(filepath):
            img = Image.open(filepath)
            ori, img = preprocessing(img)
            pred = self.model(img)
            if self.model.deep_supervision:
                pred = pred[0]
            img = color(ori, pred)
            filename, _ = os.path.splitext(filepath[filepath.rfind('/'):])
            Image.fromarray(img).save(os.path.join(output_path, "result_{}.jpg".format(filename)))

        self.model.eval()
        if os.path.isfile(path):
            single_image(path)
        else:
            for src_filename in os.listdir(path):
                single_image(os.path.join(path, src_filename))

