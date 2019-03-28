import torch.utils.data as data
import torchvision.transforms as transforms
import os
from imgaug import augmenters as iaa
import imageio
import random
import numpy as np
import torch

class ADE20K(data.Dataset):

    NBR_CLASSES = 150
    def __init__(self, mode='train', image_size=384, data_path='./data/'):
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.data_path = data_path
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

        if self.mode == 'train':
            img, mask = self.__preprocessing_for_train(img, mask)
        elif self.mode == 'val':
            img, mask = self.__preprocessing_for_validation(img, mask)

        return self.im_transform(img), self.__mask_transform(mask)

    def __preprocessing_for_validation(self, img, mask):
        h, w = img.shape
        if h > w:
            img = iaa.Resize({"height": "keep-aspect-ratio", "width": self.crop_size},
                             interpolation='linear').augment_image(img)
            mask = iaa.Resize({"height": "keep-aspect-ratio", "width": self.crop_size},
                              interpolation='nearest').augment_image(mask)
        else:
            img = iaa.Resize({"height": self.crop_size, "width": "keep-aspect-ratio"},
                             interpolation='linear').augment_image(img)
            mask = iaa.Resize({"height": self.crop_size, "width": "keep-aspect-ratio"},
                              interpolation='nearest').augment_image(mask)

        h, w = img.shape
        x = int((w - self.crop_size) // 2)
        y = int((h - self.crop_size) // 2)
        img = img[y: y + self.crop_size, x: x + self.crop_size, :]
        mask = mask[y: y + self.crop_size, x: x + self.crop_size]

        return img, mask

    def __preprocessing_for_train(self, img, mask):
        # random left-right flip
        if random.random() < 0.5:
            img = iaa.Fliplr(1.0).augment_images(img)
            mask = iaa.Fliplr(1.0).augment_images(mask)

        # random up-down flip
        if random.random() < 0.5:
            img = iaa.Flipud(1.0).augment_image(img)
            mask = iaa.Flipud(1.0).augment_image(mask)

        # random gaussian blur
        if random.random() < 0.5:
            img = iaa.GaussianBlur(sigma=(0.0, 2.0)).augment_image(img)

        # random resize
        resize_rate = random.random() * (2 - 0.5) + 0.5
        h, w = img.shape
        if h > w:
            img = iaa.Resize({"height": int(self.image_size * resize_rate), "width": "keep-aspect-ratio"},
                             interpolation='linear').augment_image(img)
            mask = iaa.Resize({"height": int(self.image_size * resize_rate), "width": "keep-aspect-ratio"},
                              interpolation='nearest').augment_image(mask)
        else:
            img = iaa.Resize({"height": "keep-aspect-ratio", "width": int(self.image_size * resize_rate)},
                             interpolation='linear').augment_image(img)
            mask = iaa.Resize({"height": "keep-aspect-ratio", "width": int(self.image_size * resize_rate)},
                              interpolation='nearest').augment_image(mask)

        # padding
        h, w = img.shape
        if min(h, w) < self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            img = iaa.CropAndPad(px=(0, pad_w, pad_h, 0), pad_mode="constant", keep_size=False).augment_image(img)
            mask = iaa.CropAndPad(px=(0, pad_w, pad_h, 0), pad_mode="constant", keep_size=False).augment_image(mask)

        # crop
        h, w = img.shape
        crop_w = random.randint(0, w - self.crop_size)
        crop_h = random.randint(0, h - self.crop_size)
        img = img[crop_h: crop_h + self.crop_size, crop_w: crop_w + self.crop_size, :]
        mask = mask[crop_h: crop_h + self.crop_size, crop_w: crop_w + self.crop_size]

        # random rotation
        if random.random() < 0.5:
            rotation_degree = random.randint(-10, 10)
            img = iaa.Affine(rotate=rotation_degree).augment_image(img)
            mask = iaa.Affine(rotate=rotation_degree, order=0).augment_image(mask)

        return img, mask

    def __mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return torch.from_numpy(target)
