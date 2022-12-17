import os
import math
import argparse
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model_resnet26 import resnet26 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
from pytorchtools import EarlyStopping
class Logger(object):
    def __init__(self, filename='./log/resnet26', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_model='resnet26'
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights/resnet26") is False:
        os.makedirs("./weights/resnet26")
    if os.path.exists("./log") is False:
        os.makedirs("./log")
    sys.stdout = Logger(stream=sys.stdout)
    # 0，1，2，3，4 分别对应类别 中间星系，雪茄星系，侧向星系，圆形星系，漩涡星系
    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(
        args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.CenterCrop(256),
                                     transforms.RandomRotation((-25, 25 )),
                                     transforms.RandomResizedCrop(224, scale=(0.9, 1)),
                                     transforms.Resize((64, 64)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065])
                                     ]),
        "test": transforms.Compose([transforms.CenterCrop(224),
                                   transforms.Resize((64, 64)),
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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if args.os == 'windows':
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
    patience = 25 #25个epoch内验证精度不下降
    early_stopping= EarlyStopping(patience, verbose=True)
    model = create_model(num_classes=args.num_classes).to(device)
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
    for epoch in range(1000):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        val_acc, val_loss= evaluate(model=model,
                           data_loader=val_loader,
                           device=device)

        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], val_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > best_val_acc:
            best_model = model
            best_val_acc=val_acc
            torch.save(best_model.state_dict(), "./weights/resnet26/bestmodel-{}.pth".format(num_model, epoch))

        print("[epoch {}] val_acc: {}  best_acc:{}".format(epoch, round(val_acc, 3),round(best_val_acc, 3)))
        early_stopping(val_loss, val_acc, model)
        if early_stopping.early_stop:
            print("epoch = {}".format(epoch))
            break
        # torch.save(model.state_dict(), "./weights/efficientnet-{}/model-{}.pth".format(num_model, epoch))
    # test
    test_acc, sum_num = evaluate(model=best_model,
                                data_loader=test_loader,
                                device=device)
    print("best test accuracy: {}".format(round(test_acc, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str,
                        default=r"F:\dataSet\clean gzdata")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--os', default='windows', help='windows or linux')

    opt = parser.parse_args()

    main(opt)
