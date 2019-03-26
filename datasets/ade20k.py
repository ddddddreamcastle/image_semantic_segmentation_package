import torch.utils.data as data
import torchvision.transforms as transforms
import os
import imgaug
from imgaug import augmenters as iaa
import imageio
import random

class ADE20K(data.Dataset):
    def __init__(self, mode='train', image_size=384, data_path='./data/', dataloader_workers=4):
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.data_path = data_path
        self.workers = dataloader_workers
        self.image_size = image_size * 1.083
        self.crop_size = image_size
        self.init_data()
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean, std=self.std
            )
        ])

    def __get_pairs(self, img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            base_filename = os.path.splitext(filename)
            mask_filename = '{}.png'.format(base_filename)
            if os.path.isfile(os.path.join(mask_folder, mask_filename)):
                img_paths.append(os.path.join(img_folder, filename))
                mask_paths.append(os.path.join(mask_folder, mask_filename))
            else:
                raise RuntimeError('cannot find the mask image for {}'.format(filename))
        return img_paths, mask_paths

    def init_data(self):
        if self.mode == 'train':
            img_folder = os.path.join(self.data_path, 'train/images')
            mask_folder = os.path.join(self.data_path, 'train/labels')
            self.images, self.masks = self.__get_pairs(img_folder, mask_folder)
        elif self.mode == 'val':
            img_folder = os.path.join(self.data_path, 'val/images')
            mask_folder = os.path.join(self.data_path, 'val/labels')
            self.images, self.masks = self.__get_pairs(img_folder, mask_folder)
        else:
            train_img_folder = os.path.join(self.data_path, 'train/images')
            train_mask_folder = os.path.join(self.data_path, 'train/labels')
            train_img_paths, train_mask_paths = self.__get_pairs(train_img_folder, train_mask_folder)
            val_img_folder = os.path.join(self.data_path, 'val/images')
            val_mask_folder = os.path.join(self.data_path, 'val/labels')
            val_img_paths, val_mask_paths = self.__get_pairs(val_img_folder, val_mask_folder)
            self.images = train_img_paths + val_img_paths
            self.mass = train_mask_paths + val_mask_paths

        if len(self.images) != len(self.masks):
            raise RuntimeError("the number of images and masks does not matching")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = imageio.imread(self.images[index])
        mask = imageio.imread(self.masks[index])

        # random left-right flip
        if random.random() < 0.5:
            img = iaa.Fliplr(1.0).augment_images(img)
            mask = iaa.Fliplr(1.0).augment_images(mask)
        # random up-down flip
        if random.random() < 0.5:
            img = iaa.Flipud(1.0).augment_image(img)
            mask = iaa.Flipud(1.0).augment_image(mask)



        return img, mask


    def transform(self, img, lbl):
        new_img = self.im_transform(img)
        return new_img, lbl

def train_loader_cubs(batch_size, num_workers=4, pin_memory=False, transform=True, shuffle=True):
    return data.DataLoader(CHINESEFOODNET(transform=transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def test_loader_cubs(batch_size, num_workers=4, pin_memory=False, transform=True, shuffle=True):
    return data.DataLoader(CHINESEFOODNET(mode='val', transform=transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
