# -*- coding: utf-8 -*-
# @Author: Arunabh Sharma
# @Date:   2024-02-19 23:14:58
# @Last Modified by:   Arunabh Sharma
# @Last Modified time: 2024-02-19 23:21:21


# Basic quadratic autograd example
import torch


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


# Basic curve fit

if __name__ == "__main__":
    quadratic_function()
