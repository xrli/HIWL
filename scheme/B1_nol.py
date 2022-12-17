import os
import math
import argparse
import sys
import copy

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from model_efficientnet import efficientnet_b1
from my_dataset import MyDataSet, returnDataset
from utils import read_split_data, train_one_epoch, evaluateall, evaluate
from pytorchtools import EarlyStopping

class Logger(object):
    def __init__(self, filename='./log/b1', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1-nol": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = args.num_model

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=.", view at http://localhost:6006/')
    tb_writer2 = SummaryWriter('classes2/{}'.format(num_model))
    tb_writer4 = SummaryWriter('classes4/{}'.format(num_model))


    if num_model == 'B0':
        create_model = efficientnet_b0
    if num_model == 'B1-nol':
        create_model = efficientnet_b1
    if num_model == 'B2':
        create_model = efficientnet_b2
    if num_model == 'B3':
        create_model = efficientnet_b3
    if os.path.exists("./weights/{}".format(num_model)) is False:
        os.makedirs("./weights/{}".format(num_model))
    if os.path.exists("./log") is False:
        os.makedirs("./log")
    if os.path.exists("./predicts/{}".format(num_model)) is False:
        os.makedirs("./predicts/{}".format(num_model))
    sys.stdout = Logger(filename='./log/efficientnet-{}'.format(num_model), stream=sys.stdout)
    # 0，1，2，3，4 分别对应类别 中间星系，雪茄星系，侧向星系，圆形星系，漩涡星系
    train_images_path5, train_images_label5, val_images_path5, val_images_label5, test_images_path5, test_images_label5 = read_split_data(args.data_path5)

    data_transform = {
        "train": transforms.Compose([transforms.CenterCrop(256),
                                     transforms.RandomRotation((-25, 25 )),
                                     transforms.RandomResizedCrop(img_size[num_model], scale=(0.9, 1)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065])
                                     ]),
        "test": transforms.Compose([transforms.CenterCrop(img_size[num_model]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065])
                                    ])}

    val_dataset5, test_dataset5, train_dataset4, val_dataset4, train_dataset2, val_dataset2 = returnDataset(data_transform, train_images_path5, train_images_label5, val_images_path5, val_images_label5, test_images_path5, test_images_label5)

    # 权重采样，定义每个类别采样的权重
    target = train_dataset2.images_class
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    # 在DataLoader的时候传入采样器即可
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if args.os == 'windows':
        nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    val_loader5 = torch.utils.data.DataLoader(val_dataset5,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw)
    test_loader5 = torch.utils.data.DataLoader(test_dataset5,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw)

    train_loader4 = torch.utils.data.DataLoader(train_dataset4,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=True,
                                                  num_workers=nw)
    val_loader4 = torch.utils.data.DataLoader(val_dataset4,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw)

    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                                 batch_size=batch_size,
                                                 pin_memory=False,
                                                 shuffle=False,
                                                 sampler=sampler,
                                                 num_workers=nw)
    val_loader2 = torch.utils.data.DataLoader(val_dataset2,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=False,
                                                num_workers=nw)
    # 如果存在预训练权重则载入
    model4 = create_model(num_classes=args.num_classes4).to(device)
    model2 = create_model(num_classes=args.num_classes2).to(device)
    tags = ["loss", "accuracy", "learning_rate"]#需要画图的指标

    #加载权重4
    if args.weights4 != "":
        if os.path.exists(args.weights4):
            weights4_dict = torch.load(args.weights4, map_location=device)
            load_weights4_dict = {k: v for k, v in weights4_dict.items()
                                    if model4.state_dict()[k].numel() == v.numel()}
            print(model4.load_state_dict(load_weights4_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights4 file: {}".format(args.weights4))



    pg4 = [p for p in model4.parameters() if p.requires_grad]
    # optimizer4 = optim.Adam(pg4, lr=args.lr4, weight_decay=1E-4)
    optimizer4 = optim.SGD(pg4, lr=args.lr4, momentum=0.9, weight_decay=1E-4)
    scheduler4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer4 , T_max=150, eta_min=0)
    # lf4 = lambda x: ((1 + math.cos(x * math.pi / args.epochs4)) / 2) * (1 - args.lrf4) + args.lrf4  # cosine
    # scheduler4 = lr_scheduler.LambdaLR(optimizer4, lr_lambda=lf4)

    # 加载权重2
    if args.weights2 != "":
        if os.path.exists(args.weights2):
            weights2_dict = torch.load(args.weights2, map_location=device)
            load_weights2_dict = {k: v for k, v in weights2_dict.items()
                                   if model2.state_dict()[k].numel() == v.numel()}
            print(model2.load_state_dict(load_weights2_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights2 file: {}".format(args.weights2))


    pg2 = [p for p in model2.parameters() if p.requires_grad]
    # optimizer2 = optim.Adam(pg2, lr=args.lr2, weight_decay=1E-4)
    optimizer2 = optim.SGD(pg2, lr=args.lr2, momentum=0.9, weight_decay=1E-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=150, eta_min=0)
    # lf2 = lambda x: ((1 + math.cos(x * math.pi / args.epochs2)) / 2) * (1 - args.lrf2) + args.lrf2  # cosine
    # scheduler2 = lr_scheduler.LambdaLR(optimizer2, lr_lambda=lf2)

    best_acc_4 = [0, 0, 0]  # 精度前三
    best_model4 = [copy.deepcopy(model4), copy.deepcopy(model4), copy.deepcopy(model4)]
    best_acc_2 = [0, 0, 0]
    best_model2 = [copy.deepcopy(model2), copy.deepcopy(model2), copy.deepcopy(model2)]
    acc_combine_best = 0
    acc_combine_best_index = 0

    patience = 25 #20个epoch内验证精度不下降
    early_stopping4 = EarlyStopping(patience, verbose=True)
    early_stopping2 = EarlyStopping(patience, verbose=True)
    for epoch in range(1000):
        mean_loss4 = train_one_epoch('model4', model=model4,
                                       optimizer=optimizer4,
                                       data_loader=train_loader4,
                                       device=device,
                                       epoch=epoch)

        scheduler4.step()
        # validate

        acc_4, val_loss_4 = evaluate(model=model4,
                           data_loader=val_loader4,
                           device=device)
        if acc_4 > best_acc_4[2] and acc_4 < best_acc_4[1]:
            best_acc_4[2] = acc_4
            best_model4[2] = copy.deepcopy(model4)
            torch.save(model4.state_dict(), "./weights/{}/best_model4_2.pth".format(num_model))
        elif acc_4 > best_acc_4[1] and acc_4 < best_acc_4[0]:
            best_acc_4[2] = best_acc_4[1]
            best_model4[2] = best_model4[1]
            best_acc_4[1] = acc_4
            best_model4[1] = copy.deepcopy(model4)
            torch.save(model4.state_dict(), "./weights/{}/best_model4_1.pth".format(num_model))
        elif acc_4 > best_acc_4[0]:
            best_acc_4[2] = best_acc_4[1]
            best_model4[2] = best_model4[1]
            best_acc_4[1] = best_acc_4[0]
            best_model4[1] = best_model4[0]
            best_acc_4[0] = acc_4
            best_model4[0] = copy.deepcopy(model4)
            torch.save(model4.state_dict(), "./weights/{}/best_model4_0.pth".format(num_model))

        tb_writer4.add_scalar(tags[0], mean_loss4, epoch)
        tb_writer4.add_scalar(tags[1], acc_4, epoch)
        tb_writer4.add_scalar(tags[2], optimizer4.param_groups[0]["lr"], epoch)
        print("epoch {}, acc4: {}, best_acc_4: {}".format(epoch, acc_4, best_acc_4[0]))
        early_stopping4(val_loss_4, acc_4, model4)
        if early_stopping4.early_stop:
            break
    for epoch in range(1000):
        mean_loss2 = train_one_epoch('model2', model=model2,
                                      optimizer=optimizer2,
                                      data_loader=train_loader2,
                                      device=device,
                                      epoch=epoch)
        scheduler2.step()

        acc_2, val_loss_2 = evaluate(model=model2,
                          data_loader=val_loader2,
                          device=device)
        if acc_2 > best_acc_2[2] and acc_2 < best_acc_2[1]:
            best_acc_2[2] = acc_2
            best_model2[2] = copy.deepcopy(model2)
            torch.save(model2.state_dict(), "./weights/{}/best_model2_2.pth".format(num_model))
        elif acc_2 > best_acc_2[1] and acc_2 < best_acc_2[0]:
            best_acc_2[2] = best_acc_2[1]
            best_model2[2] = best_model2[1]
            best_acc_2[1] = acc_2
            best_model2[1] = copy.deepcopy(model2)
            torch.save(model2.state_dict(), "./weights/{}/best_model2_1.pth".format(num_model))
        elif acc_2 > best_acc_2[0]:
            best_acc_2[2] = best_acc_2[1]
            best_model2[2] = best_model2[1]
            best_acc_2[1] = best_acc_2[0]
            best_model2[1] = best_model2[0]
            best_acc_2[0] = acc_2
            best_model2[0] = copy.deepcopy(model2)
            torch.save(model2.state_dict(), "./weights/{}/best_model2_0.pth".format(num_model))
        tb_writer2.add_scalar(tags[0], mean_loss2, epoch)
        tb_writer2.add_scalar(tags[1], acc_2, epoch)
        tb_writer2.add_scalar(tags[2], optimizer2.param_groups[0]["lr"], epoch)
        print("epoch {}, acc2: {}, best_acc_2: {}".format(epoch, acc_2, best_acc_2[0]))
        early_stopping2(val_loss_2, acc_2, model2)
        if early_stopping2.early_stop:
            print("epoch = {}".format(epoch))
            break
    #验证总的
    for i in range(len(best_model2)):
        for j in range(len(best_model4)):

            acc_combine, pred_all = evaluateall(
                model2=best_model2[i],
                model4=best_model4[j],
                test_loader5=val_loader5,
                device=device)
            torch.save(pred_all, './predicts/{}/pred_all-{}-{}.pth'.format(num_model, i, j))
            if acc_combine_best < acc_combine:
                acc_combine_best = acc_combine
                acc_combine_best_index = (i, j)

    test_acc, test_pred_all = evaluateall(
        model2=best_model2[acc_combine_best_index[0]],
        model4=best_model4[acc_combine_best_index[1]],
        test_loader5=test_loader5,
        device=device)
    torch.save(test_pred_all, './predicts/{}/test_pred_all.pth'.format(num_model))
    print("acc_combine_best: {}, acc_combine_best_index: {}, test_acc: {}".format(acc_combine_best, acc_combine_best_index, test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path5', type=str,
                        default=r"F:\dataSet\clean gzdata")
    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--num_classes4', type=int, default=4)
    parser.add_argument('--epochs4', type=int, default=100)
    parser.add_argument('--lr4', type=float, default=0.005)
    parser.add_argument('--lrf4', type=float, default=0.05)
    parser.add_argument('--weights4', type=str, default='F:/pretrain pth/efficientnetb1.pth',
                        help='initial weights4 path')

    parser.add_argument('--num_classes2', type=int, default=2)
    parser.add_argument('--epochs2', type=int, default=150)
    parser.add_argument('--lr2', type=float, default=0.005)
    parser.add_argument('--lrf2', type=float, default=0.01)
    parser.add_argument('--weights2', type=str, default='F:/pretrain pth/efficientnetb1.pth',
                        help='initial weights2 path')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--os', default='windows', help='windows or linux')
    parser.add_argument('--num-model', default='B1-nol', help='B0-B7')

    opt = parser.parse_args()

    main(opt)
