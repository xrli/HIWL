import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
# 标签平滑嵌入到loss函数
class SMLoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            # 实现
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()

def read_split_data(root: str, val_rate: float = 0.1, test_rate: float = 0.1):
    split_rate = val_rate + test_rate
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    galaxy_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    split_galaxy_class = []  # 存储切分后类别
    for i in galaxy_class:
        split_galaxy_class.append(i + '_train')
        split_galaxy_class.append(i + '_test')
    # 排序，保证顺序一致
    galaxy_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(galaxy_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    split_every_class_num = []  # 存储每个类别的切分后样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in galaxy_class:
        cla_path = os.path.join(root, cla)
        sample_count = 0
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        split_path = random.sample(images, round(len(images) * split_rate))
        for img_path in images:
            if img_path in split_path:  # 如果该路径在采样的集合样本中则存入划分集
                sample_count += 1
                if sample_count <= len(split_path)*(val_rate/split_rate):
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:
                    test_images_path.append(img_path)
                    test_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for val.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(split_every_class_num)), split_every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(split_every_class_num)),split_galaxy_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(split_every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('galaxy class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(str,model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, targets = data
        pred = model(images.to(device))
        loss = loss_function(pred, targets.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] {}-meanloss: {}".format(epoch, str, round(mean_loss.item(), 3))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    print('traloss: {}'.format(mean_loss.item()))
    return mean_loss.item()

def train_googlenet_one_epoch(str, model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    # 标签平滑
    loss_function = SMLoss(label_smooth=0.05, class_num=int(str.split('l')[1]))
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)


    for step, data in enumerate(data_loader):
        images, labels = data

        logits, aux_logits2, aux_logits1 = model(images.to(device))
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    print('traloss: {}'.format(mean_loss.item()))
    return mean_loss.item()

@torch.no_grad()
def evaluateall(model2, model4,test_loader5, device):
    model2.eval()
    model4.eval()
    classes2=[1, 2]
    classes4=[0, 1, 3, 4]
    # 验证样本总个数
    total_num = len(test_loader5.dataset)
    correct_num=torch.tensor([0]).to(device)#统计预测正确个数
    pred_all=torch.tensor([]).to(device)#不断拼接最终输出为预测列表
    for step, data in enumerate(tqdm(test_loader5)):
        images5, labels5= data
        pred4 = model4(images5.to(device))
        pred4 = torch.max(pred4, dim=1)[1]#3类的预测
        preddata=copy.deepcopy(pred4)#每个batch的预测结果
        preddata=torch.tensor([classes4[i] for i in preddata.tolist()])#将索引转为对应类别

        select_index2=torch.where(preddata==1)#选出为2类的索引

        if len(select_index2[0])!=0:
            select_images2=images5[select_index2]#选出为2类的图片
            pred2=model2(select_images2.to(device))
            pred2=torch.max(pred2,dim=1)[1]
            pred2 = torch.tensor([classes2[i] for i in pred2.tolist()])
            for i,j in enumerate(select_index2[0].tolist()):
                preddata[j]=pred2[i]
        correct_num += torch.eq(preddata.to(device), labels5.to(device)).sum()
        pred_all = torch.cat([pred_all, preddata.to(device)], dim=0)

    acc_combine = correct_num.item() / total_num
    return acc_combine,pred_all

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    loss_function = torch.nn.CrossEntropyLoss()
    sum_num = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader)
    mean_loss = torch.zeros(1).to(device)
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        pred_label = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred_label, labels.to(device)).sum()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
    acc = sum_num.item() / total_num

    return acc, mean_loss.item()






