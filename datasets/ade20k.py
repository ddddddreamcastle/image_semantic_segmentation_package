import torch.utils.data as data
import torchvision.transforms as transforms
import os

class ADE20K(data.Dataset):
    def __init__(self, mode='train', image_size=384):
        self.mode = mode
        self.mean = [0.553, 0.478, 0.364]
        self.std = [0.262, 0.269, 0.288]
        self.data = []
        self.data_base_dir = os.path.join(
            '/DATA1000/zhaopeng/workspace/research/dishes_recognition/data/chinesefoodnet',
            'train_noise' if self.mode == 'train' else self.mode)
        self.init_data()
        self.im_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean, std=self.std
            )
        ])


    def init_data(self):
        if self.mode == 'train':
            datalist = pickle.load(open(
                '/DATA1000/zhaopeng/workspace/research/dishes_recognition/data/chinesefoodnet/simple_train.pkl',
                'rb'))
        else:
            datalist = pickle.load(
                open('/DATA1000/zhaopeng/workspace/research/dishes_recognition/data/chinesefoodnet/simple_val.pkl', 'rb'))

        random.shuffle(datalist)
        for idx, line in enumerate(datalist):
            line = line.strip().split(' ')
            path = os.path.join(self.data_base_dir, line[0])
            label = line[-1]
            self.data.append((path, int(label)))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        path, label = self.data[index]

        img = Image.open(path)

        img_ndarr = np.asarray(img)
        if img_ndarr.shape[2] != 3:
            img_ndarr = img_ndarr[:,:,0:3]
            img = Image.fromarray(img_ndarr)

        if img.mode == 'L':
            img = img.convert('RBG')

        # img = scale_byRatio(path, return_width=IMSIZE)
        # img = img / 255.

        if self._transform:
            return self.transform(img, label)

        return img, label


    def transform(self, img, lbl):
        new_img = self.im_transform(img)
        return new_img, lbl

def train_loader_cubs(batch_size, num_workers=4, pin_memory=False, transform=True, shuffle=True):
    return data.DataLoader(CHINESEFOODNET(transform=transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def test_loader_cubs(batch_size, num_workers=4, pin_memory=False, transform=True, shuffle=True):
    return data.DataLoader(CHINESEFOODNET(mode='val', transform=transform), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
