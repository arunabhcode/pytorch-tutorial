# -*- coding: utf-8 -*-
# @Author: Arunabh Sharma
# @Date:   2024-02-19 23:14:58
# @Last Modified by:   Arunabh Sharma
# @Last Modified time: 2024-02-21 23:16:35


# Basic quadratic autograd example
import torch
import numpy as np
import torchvision
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quadratic_function():
    x = torch.tensor(8.0, requires_grad=True)
    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    c = torch.tensor(1.0, requires_grad=True)

    y = a * x**2 + b * x + c

    y.backward()

    print(x.grad)
    print(a.grad)
    print(b.grad)
    print(c.grad)


# Basic quadratic curve fit
class Quadratic(torch.nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()

        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c

    def string(self):
        return f"y = {self.a.item()} x^2 + {self.b.item()} x + {self.c.item()}"


def quadratic_curve_fit():
    radius = 5
    x = torch.linspace(-radius, radius, 1000)
    rand_idx = torch.randperm(x.shape[0])
    x = x[rand_idx]
    y = radius**2 - x**2

    model = Quadratic()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    convergence = 1e-2
    iter = 0
    while True:
        iter += 1
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if iter % 100 == 0:
            print(f"Iteration: {iter}, Loss: {loss.item()}")
            print(model.string())
            print("dL/da: ", model.a.grad)
            print("dL/db: ", model.b.grad)
            print("dL/dc: ", model.c.grad)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < convergence:
            break


def numpy_torch_convert():
    x = np.array([[1, 2], [3, 4]])
    y = torch.from_numpy(x)
    z = y.numpy()
    print(x)
    print(y)
    print(np.equal(x, z).all())


def torch_dataset_loader(img_display=False):
    train_dataset = torchvision.datasets.MNIST(
        root="../../data/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="../../data/",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True
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


def transfer_learning(train_loader, model_save=True):
    resnet = torchvision.models.resnet18(pretrained=True)
    # If you want to finetune only the top layer of the model, set as below.
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace the top layer for finetuning.
    resnet.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    print(resnet)

    resnet = resnet.to(device)

    epoch = 5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters())

    for i in range(epoch):
        data_iter = iter(train_loader)
        while (ret := next(data_iter, None)) is not None:
            images, labels = ret[0], ret[1]
            images = images.to(device)
            labels = labels.to(device)
            loss = criterion(resnet(images), labels)

            print(f"Iteration: {i}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if model_save:
        torch.save(resnet, "model.ckpt")

    return resnet


def explore_transfer_learning(test_loader, model_name):
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
            # image_r = image.reshape(-1, 28 * 28)
            # pred_label = model(image_r)
            pred_label = model(image)
            img = image[0].numpy().transpose(1, 2, 0)
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


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc0 = torch.nn.Linear(28 * 28, 128)
        self.nl1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(128, 10)

    def forward(self, x):
        return self.fc1(self.nl1(self.fc0(x)))


def train_nn(train_loader, model_save=True):
    model = NeuralNetwork().to(device)

    epoch = 1
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(epoch):
        data_iter = iter(train_loader)
        while (ret := next(data_iter, None)) is not None:
            images, labels = ret[0], ret[1]
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            loss = criterion(model(images), labels)

            print(f"Iteration: {i}, Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if model_save:
        torch.save(model, "nn.ckpt")

    return model


if __name__ == "__main__":
    train_loader, test_loader = torch_dataset_loader(False)
    _ = transfer_learning(train_loader)
    explore_transfer_learning(test_loader, "model.ckpt")
    # explore_transfer_learning(test_loader, "nn.ckpt")
