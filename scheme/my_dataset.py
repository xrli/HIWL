from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import copy

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):

        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def returnDataset(data_transform, train_images_path5, train_images_label5, val_images_path5, val_images_label5, test_images_path5, test_images_label5):
    #分层学习标签调整
    # 两类合成一类后剩下4类[0,1,3,4],调整标签为[0,1,2,3]
    train_images_path4 = copy.deepcopy(train_images_path5)
    train_images_label4 = copy.deepcopy(train_images_label5)
    val_images_path4 = copy.deepcopy(val_images_path5)
    val_images_label4 = copy.deepcopy(val_images_label5)

    for i in range(len(train_images_label4)):  # 标签2改1，3改2,4改3
        if (train_images_label4[i] == 2):
            train_images_label4[i] = 1
        if (train_images_label4[i] == 3):
            train_images_label4[i] = 2
        if (train_images_label4[i] == 4):
            train_images_label4[i] = 3
    for i in range(len(val_images_label4)):  # 标签2改1，3改2,4改3
        if (val_images_label4[i] == 2):
            val_images_label4[i] = 1
        if (val_images_label4[i] == 3):
            val_images_label4[i] = 2
        if (val_images_label4[i] == 4):
            val_images_label4[i] = 3

    # 雪茄类和侧向类的训练测试集[1,2],调整标签为[0,1]
    train_images_label2 = (
            np.array(copy.deepcopy(train_images_label5))[
                (np.array(train_images_label5) == 1) | (np.array(train_images_label5) == 2)]-1).tolist()  # 全部减1
    val_images_label2 = (
            np.array(copy.deepcopy(val_images_label5))[
                (np.array(val_images_label5) == 1) | (np.array(val_images_label5) == 2)]-1).tolist()  # 全部减1
    train_images_path2 = (
        np.array(copy.deepcopy(train_images_path5))[
            (np.array(train_images_label5) == 1) | (np.array(train_images_label5) == 2)]).tolist()
    val_images_path2 = (
        np.array(copy.deepcopy(val_images_path5))[
            (np.array(val_images_label5) == 1) | (np.array(val_images_label5) == 2)]).tolist()


    # 实例化数据集

    val_dataset5 = MyDataSet(images_path=val_images_path5,
                              images_class=val_images_label5,
                              transform=data_transform["test"])
    test_dataset5 = MyDataSet(images_path=test_images_path5,
                              images_class=test_images_label5,
                              transform=data_transform["test"])
    train_dataset4 = MyDataSet(images_path=train_images_path4,
                                 images_class=train_images_label4,
                                 transform=data_transform["train"])
    val_dataset4 = MyDataSet(images_path=val_images_path4,
                                images_class=val_images_label4,
                                transform=data_transform["test"])

    train_dataset2 = MyDataSet(images_path=train_images_path2,
                                images_class=train_images_label2,
                                transform=data_transform["train"])
    val_dataset2 = MyDataSet(images_path=val_images_path2,
                               images_class=val_images_label2,
                               transform=data_transform["test"])
    return val_dataset5, test_dataset5, train_dataset4, val_dataset4, train_dataset2, val_dataset2