from models.nn import get_model
from datasets import get_train_val_dataset
import torch

class Manager(object):
    def __init__(self, args):
        kwargs = vars(args)
        self.model = get_model(name=args.model, kwargs=kwargs)
        self.train_loader, self.val_loader = get_train_val_dataset('ade20k', kwargs)

        parameters = self.model.get_parameters_as_groups(args.lr)

        self.optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                          weight_decay=args.weight_decay)

        self.loss =

    def __do_batch(self):
        pass

    def __do_epoch(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass