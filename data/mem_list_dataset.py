from torch.utils.data import Dataset
import cv2


class MemListDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super(self, MemListDataset).__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list
        infos = [line.split() for line in open(data_list).readlines()]
        img_paths = [info[0] for info in infos]
        label_paths = [infos[1] for info in infos]
        self.imgs = []
        self.labels = []
        for img_path, label_path in zip(img_paths, label_paths):
            img = cv2.imread(img_path)
            label = cv2.imread(label_path)
            # if you have large memory, you can load all images into memory to accelerate training
            img, label = self.preprocess(img, label)
            self.imgs.append(img)
            self.labels.append(label)

    def preprocess(self, img, label):
        # cv: h, w, c, tensor: c, h, w
        img = img.transpose((2, 0, 1)).astype(np.float32)
        # you can add other process method or augment here
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label
