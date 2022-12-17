import os
import math
import argparse
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from model_efficientnet import efficientnet_b1 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate_noh

class Logger(object):
    def __init__(self, filename='./log/effnoh', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter()
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B1_noh"
    if os.path.exists("./weights/{}".format(num_model)) is False:
        os.makedirs("./weights/{}".format(num_model))
    if os.path.exists("../scheme/log") is False:
        os.makedirs("./log")

    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.CenterCrop(256),
                                     transforms.RandomRotation((-25, 25 )),
                                     transforms.RandomResizedCrop(img_size['B1'], scale=(0.9, 1)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065])
                                     ]),
        "test": transforms.Compose([transforms.CenterCrop(img_size['B1']),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065])
                                    ])}
    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["test"])
    # 实例化测试数据集
    test_data_set = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])
    batch_size = args.batch_size
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    # model.load_state_dict(torch.load(args.weights, map_location=device))
    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_val_acc =0.0
    best_model=None
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        sum_num, pred_all = evaluate_noh(model=model,
                           data_loader=val_loader,
                           device=device)

        val_acc = sum_num / len(val_data_set)
        torch.save(model.state_dict(), "./weights/B1_noh/model-{}.pth".format(epoch))
        if val_acc > best_val_acc:
            best_model = model
            best_val_acc=val_acc
            torch.save(pred_all, 'b1_noh_val_811.pth')
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], val_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

    # test
    sum_num, pred_all = evaluate_noh(model=model,
                                     data_loader=test_loader,
                                     device=device)
    test_acc = sum_num / len(test_data_set)
    torch.save(pred_all, 'b1_noh_test_811.pth')
    print('test_acc:{}'.format(test_acc))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str,
                        default=r"F:/dataSet/clean gzdata")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lrf', type=float, default=0.005)
    parser.add_argument('--weights', type=str, default=r'',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
