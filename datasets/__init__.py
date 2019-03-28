import torch.utils.data as data
from .ade20k import ADE20K

datasets = {
        'ade20k': ADE20K
    }

def get_train_val_loader(name, image_size=384, data_path='./data/', batch_size=8, shuffle=True, num_workers=4,
                         pin_memory=False, **kwargs):
    global datasets
    train_loader = data.DataLoader(datasets[name](mode='train', image_size=image_size, data_path=data_path),
                                   batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                   pin_memory=pin_memory)
    val_loader = data.DataLoader(datasets[name](mode='val', image_size=image_size, data_path=data_path),
                                 batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=pin_memory)
    return train_loader, val_loader