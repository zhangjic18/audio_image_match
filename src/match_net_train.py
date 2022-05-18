import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.match_net import match_net

from dataset.match_dataset import match_dataset
from utils import AverageMeter


def main(args):
    device = torch.device("cuda:" + args.device[0] if torch.cuda.is_available() else "cpu")
    model = match_net(device)

    if len(args.device) > 1:  # 指定多块GPU进行训练
        model = nn.DataParallel(model, device_ids=[int(item) for item in args.device.split(",")])

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == "SGDm":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

    for i in range(args.episode):
        # 重新构建数据集
        total_dataset = match_dataset()

        dataset_len_list = [math.floor(0.8 * len(total_dataset)),
                            math.floor(0.1 * len(total_dataset)),
                            math.floor(0.1 * len(total_dataset)),
                            ]
        dataset_len_list[0] += (len(total_dataset) - sum(dataset_len_list))

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(total_dataset, dataset_len_list)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # 1个episode，1次train与1次test
        train(i, args, model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler)

        test_accuracy = test(model, test_dataloader, device)

        print("====================================episode " + str(i) + ":test====================================")
        print("test_accuracy: {:.6f}".format(test_accuracy))


def train(i, args, model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler):
    print("====================================episode " + str(i) + ":train====================================")
    model.train()

    epoch_loss = AverageMeter()
    best_val_epoch = -1
    best_val_accuracy = 0

    for epoch in range(args.epoch):
        correct, total = 0, 0
        epoch_loss.reset()

        for image, audio, label in train_dataloader:
            image = image.to(device)
            audio = audio.to(device)
            label = label.to(device)

            output = model(image, audio)

            loss = criterion(output, label)

            assert loss.item() != np.nan

            epoch_loss.update(loss.item(), image.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(output, dim=1) == label).sum().item()
            total += label.shape[0]

        val_accuracy = test(model, val_dataloader, device)
        if val_accuracy > best_val_accuracy:
            torch.save(model, "../checkpoints/match_net.pth")
            best_val_accuracy = val_accuracy
            best_val_epoch = epoch

        if args.scheduler == "MultiStepLR":
            scheduler.step()
        else:
            scheduler.step(best_val_accuracy)

        print("epoch: {}/{}, loss: {:.6f}, train accuracy: {:.6f}, val accuracy: {:.6f}, lr: {:.6f}".format(
            epoch, args.epoch, epoch_loss.avg, correct / total, val_accuracy, optimizer.param_groups[0]['lr']))

        if epoch - best_val_epoch >= 7:
            print("stop training early at epoch ", epoch)
            break

    print("====================================episode " + str(i) + ":best val====================================")
    print("best_val_epoch: {}, best_val_accuracy: {:.6f}".format(best_val_epoch, best_val_accuracy))


def test(model, test_dataloader, device):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0

        for image, audio, label in test_dataloader:
            image = image.to(device)
            audio = audio.to(device)
            label = label.to(device)

            output = model(image, audio)

            correct += (torch.argmax(output, dim=1) == label).sum().item()
            total += label.shape[0]

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match net train")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--episode", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGDm")
    parser.add_argument("--scheduler", type=str, default="MultiStepLR")
    parser.add_argument("--device", type=str, default="6")

    args = parser.parse_args()
    main(args=args)
