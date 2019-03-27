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


    def __do_batch(self):
        pass

    def __do_epoch(self, epoch):
        pass

    def fit(self):

        for epoch in self.epochs:
            self.model.train()
            print('Epoch : {}'.format(epoch + 1))
            self.__do_epoch(epoch + 1)


    def predict(self):
        pass