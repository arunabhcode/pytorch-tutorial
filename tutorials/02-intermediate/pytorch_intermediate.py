# -*- coding: utf-8 -*-
# @Author: Arunabh Sharma
# @Date:   2024-02-21 23:38:56
# @Last Modified by:   Arunabh Sharma
# @Last Modified time: 2024-02-25 22:14:30

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_dataset_loader(img_display=False):
    train_dataset = torchvision.datasets.MNIST(
        root="../data/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="../data/",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False
    )

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(images.size())
    print(labels.size())

    img = images[0].numpy().transpose(1, 2, 0)
    img = cv2.resize(img, (800, 800))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img.shape)

    if img_display:
        cv2.imshow("images", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return train_loader, test_loader


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.fc1 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        print(x.shape)
        x = x.reshape(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)
        return x


def train_model(model, epoch, train_loader, model_name, model_save):
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(epoch):
        data_iter = iter(train_loader)
        while (ret := next(data_iter, None)) is not None:
            images, labels = ret[0], ret[1]
            # images = images.to(device)
            images = images.reshape(-1, 28, 28).to(device)
            labels = labels.to(device)
            loss = criterion(model(images), labels)

            print(f"Iteration: {i}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if model_save:
        torch.save(model, model_name)

    return model


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_model_decay_lr(model, epoch, lr, train_loader, model_name, model_save):
    model = model.to(device)

    curr_lr = lr
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr)

    for i in range(epoch):
        data_iter = iter(train_loader)
        while (ret := next(data_iter, None)) is not None:
            images, labels = ret[0], ret[1]
            images = images.to(device)
            labels = labels.to(device)
            loss = criterion(model(images), labels)

            print(f"Iteration: {i}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        curr_lr /= 3.0
        update_lr(optimizer, curr_lr)
    if model_save:
        torch.save(model, model_name)

    return model


def explore_model(test_loader, model_name):
    model = torch.load(model_name).to("cpu")
    model.eval()

    window_name = "Image"

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org0 = (30, 30)
    org1 = (30, 60)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 255, 0)

    # Line thickness of 2 px
    thickness = 2

    with torch.no_grad():
        for image, label in test_loader:
            image = image.reshape(-1, 28, 28)
            pred_label = model(image)
            img = image[0].numpy()
            img = cv2.resize(img, (800, 800))
            img = np.dstack((img, img, img))
            img = cv2.putText(
                img,
                "GT Label:" + str(int(label[0])),
                org0,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            pred_label = np.argmax(pred_label.numpy())
            img = cv2.putText(
                img,
                "Pred Label:" + str(int(pred_label)),
                org1,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, img)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                exit(0)


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResnetBlock, self).__init__()

        if downsample:
            self.conv1 = self.conv_layer(in_channels, out_channels, 3, 2)
            self.ds = self.conv_layer(in_channels, out_channels, 1, 2, 0)
        else:
            self.conv1 = self.conv_layer(in_channels, out_channels, 3, 1)
            self.ds = torch.nn.Identity()
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.nl1 = torch.nn.ReLU(inplace=True)
        self.conv2 = self.conv_layer(out_channels, out_channels, 3, 1)

    def conv_layer(self, in_channels, out_channels, kernel_size, stride, padding=1):
        return torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        out = self.conv1(x)
        residual = self.ds(x)
        out = self.bn1(out)
        out = self.nl1(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out += residual
        out = self.nl1(out)

        return out


class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.layer1 = ResnetBlock(1, 16, False)
        self.layer2 = ResnetBlock(16, 32, True)
        self.avg_pool = torch.nn.AvgPool2d(2)
        self.fc = torch.nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    train_loader, test_loader = torch_dataset_loader(False)
    model_name = "rnn.ckpt"
    model = RNN(28, 128, 2, 10)
    # _ = train_model(model, 4, train_loader, model_name, True)
    explore_model(test_loader, model_name)
