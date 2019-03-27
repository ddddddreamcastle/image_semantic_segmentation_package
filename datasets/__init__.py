from .ade20k import train_loader_ade20k, val_loader_ade20k

def get_train_val_dataset(name, **kwargs):
    """
    return a tuple (train_loader, val_loader)
    :param name: data set name
    :param kwargs: other params needed by data loader
    :return: tuple
    """
    dataset = {
        'ade20k': (train_loader_ade20k, val_loader_ade20k)
    }
    return dataset[name][0](**kwargs), dataset[name][1](**kwargs)