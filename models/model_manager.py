from models.nn import get_model
from datasets import get_train_val_dataset
import torch
from loss import SegmentationLoss
from torch import nn
from utils.learning_rate_scheduler import LearningRateScheduler
from tqdm import tqdm

class Manager(object):

    def __init__(self, args):
        kwargs = vars(args)
        self.epochs = args.epochs
        self.model = get_model(name=args.model, kwargs=kwargs)
        self.train_loader, self.val_loader = get_train_val_dataset('ade20k', kwargs)

        parameters = self.model.get_parameters_as_groups(args.lr)

        self.optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                          weight_decay=args.weight_decay)

        if args.supervision:
            self.criterion = SegmentationLoss((nn.CrossEntropyLoss, nn.CrossEntropyLoss),
                                     (1, args.supervision_weight))
        else:
            self.criterion = SegmentationLoss((nn.CrossEntropyLoss, ),
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
        for i, (image, target) in enumerate(tqdm_bar):
            cur_lr = self.lr_scheduler.adjust_learning_rate(self.optimizer, i, epoch)
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            preds = self.model(image)
            loss = self.criterion(preds, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tqdm_bar.set_description('Lr: {:.4}, Train loss: {:.4}'.format(cur_lr, train_loss/(i+1)))

    def __do_validation(self):
        tqdm_bar = tqdm(self.val_loader)
        for i, (image, target) in enumerate(tqdm_bar):
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()
            pred = self.model(image)[0]
            




    def fit(self):

        for epoch in self.epochs:
            print('Epoch : {}'.format(epoch + 1))
            # train step
            self.model.train()
            self.__do_epoch(epoch)

            # val step
            self.model.eval()
            with torch.no_grad():
                self.__do_validation()



    def predict(self):
        pass