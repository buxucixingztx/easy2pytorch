"""
基本语法
"""

import torch
import numpy as np

print(torch.__version__)


def delimiter():
    print("\n--------------------------------------\n")


# 定义矩阵
x = torch.empty(2, 2)
print(x)
delimiter()

# 定义随机初始化矩阵
x = torch.randn(2, 2)
print(x)
delimiter()

# 定义初始化为零
x = torch.zeros(3, 3)
print(x)
delimiter()

# 定义数据为tensor
x = torch.tensor([5.1, 2., 3., 1.])
print(x)
delimiter()

# 操作
a = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
b = torch.tensor([11., 12., 13., 14., 15., 16., 17., 18.])
c = a.add(b)
print(c)
delimiter()

# 维度变换 2x4
a = a.view(-1, 4)
b = b.view(-1, 4)
c = torch.add(a, b)
print(c)
print(f"a.size: {a.size()}; b.size: {b.size()}")
delimiter()

# torch转numpy
na = a.numpy()
nb = b.numpy()
print(f"na = {na};\nnb = {nb}")
delimiter()

# 操作
d = np.array([21., 22., 23., 24., 25., 26., 27., 28.], dtype=np.float32)
print(d.reshape(2, 4))
d = torch.from_numpy(d.reshape(2, 4))
sum = torch.sub(c, d)
print(f"sum = {sum};\nsum.size = {sum.size()}")
delimiter()

# using CUDA
if torch.cuda.is_available():
    result - d.cuda() + c.cuda()
    print(f"result = {result}")

# 自动梯度
x = torch.randn(1, 5, requires_grad=True)
y = torch.randn(5, 3, requires_grad=True)
z = torch.randn(3, 1, requires_grad=True)
print(f"x = {x};\ny = {y};\nz = {z};\n")
# delimiter()
xy = torch.matmul(x, y)
xyz = torch.matmul(xy, z)
xyz.backward()
print(f"x.grad: {x.grad};\ny.grad: {y.grad};\nz.grad: {z.grad}")
