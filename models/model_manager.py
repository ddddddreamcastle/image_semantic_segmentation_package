from models.nn import get_model

class Manager(object):
    def __init__(self, args):
        kwargs = vars(args)
        self.model = get_model(name=args.model, kwargs=kwargs)
        self.train_loader = None
        self.val_loader = None

    def fit(self):
        pass

    def predict(self):
        pass