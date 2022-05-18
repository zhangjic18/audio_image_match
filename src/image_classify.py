import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.from_torch.resnet import *
from model.from_torch.vgg import *
from model.my_model import my_model

from dataset.image_classify_dataset import image_classify_dataset
from utils import AverageMeter


def main(args):
    train_dataloader = DataLoader(dataset=image_classify_dataset(flag="train"), batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=image_classify_dataset(flag="val"), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=image_classify_dataset(flag="test"), batch_size=args.batch_size, shuffle=False)

    if args.model == "resnet18":
        model = resnet18(num_classes=10, in_channels=60, initial_stride=(4, 4))
    elif args.model == "resnet34":
        model = resnet34(num_classes=10, in_channels=60, initial_stride=(4, 4))
    elif args.model == "vgg11_bn":
        model = vgg11_bn(num_classes=10, in_channels=60, initial_stride=(4, 4))
    elif args.model == "vgg19_bn":
        model = vgg19_bn(num_classes=10, in_channels=60, initial_stride=(4, 4))
    elif args.model == "my_model":
        model = my_model()

    device = torch.device("cuda:" + args.device[0] if torch.cuda.is_available() else "cpu")
    if len(args.device) > 1:  # 指定多块GPU进行训练
        model = nn.DataParallel(model, device_ids=[int(item) for item in args.device.split(",")])
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == "SGDm":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

    train(args, model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler)

    test_accuracy = test(model, test_dataloader, device)

    print("========================================test========================================")
    print("test_accuracy: {:.6f}".format(test_accuracy))


def train(args, model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler):
    print("========================================train========================================")
    model.train()

    epoch_loss = AverageMeter()
    best_val_epoch = -1
    best_val_accuracy = 0

    for epoch in range(args.epoch):
        correct, total = 0, 0
        epoch_loss.reset()

        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            output = model(img)

            loss = criterion(output, label)

            assert loss.item() != np.nan

            epoch_loss.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(output, dim=1) == label).sum().item()
            total += label.shape[0]

        val_accuracy = test(model, val_dataloader, device)
        if val_accuracy > best_val_accuracy:
            torch.save(model, "../checkpoints/" + args.model + "__to_image_classify.pth")
            best_val_accuracy = val_accuracy
            best_val_epoch = epoch

        if args.scheduler == "MultiStepLR":
            scheduler.step()
        else:
            scheduler.step(best_val_accuracy)

        print("epoch: {}/{}, loss: {:.6f}, train accuracy: {:.6f}, val accuracy: {:.6f}, lr: {:.6f}".format(
            epoch, args.epoch, epoch_loss.avg, correct / total, val_accuracy, optimizer.param_groups[0]['lr']))

    print("========================================best val========================================")
    print("best_val_epoch: {}, best_val_accuracy: {:.6f}".format(best_val_epoch, best_val_accuracy))


def test(model, test_dataloader, device):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0

        for img, label in test_dataloader:
            img = img.to(device)
            label = label.to(device)

            output = model(img)

            correct += (torch.argmax(output, dim=1) == label).sum().item()
            total += label.shape[0]

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image classify")
    parser.add_argument("--model", type=str, default="my_model")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default="SGDm")
    parser.add_argument("--scheduler", type=str, default="MultiStepLR")
    parser.add_argument("--device", type=str, default="4,1")

    args = parser.parse_args()
    main(args=args)
