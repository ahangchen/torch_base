from torch.utils.data import Dataset
import cv2
import numpy as np


class ListDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super(self, ListDataset).__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list
        infos = [line.split() for line in open(data_list).readlines()]
        self.img_paths = [info[0] for info in infos]
        self.label_paths = [info[1] for info in infos]

    def preprocess(self, img, label):
        # cv: h, w, c, tensor: c, h, w
        img = img.transpose((2, 0, 1)).astype(np.float32)
        # you can add other process method or augment here
        return img, label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        label = cv2.imread(self.label_paths[idx])
        img, label = self.preprocess(img, label)
        return img, label
