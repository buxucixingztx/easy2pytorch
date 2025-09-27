"""
使用opencv加载onnx模型，并进行推理

首先保存mnist图片
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv
import cv2 as cv


def save_mnist():
    """
    保存手写数字图片
    """
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    test_ts = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_dl = DataLoader(test_ts, batch_size=2, shuffle=True, drop_last=False)

    for img, label in test_dl:
        print(img.shape)
        img1, img2 = img.numpy()
        img1 = np.reshape(img1, (28, 28)) * 255
        img2 = np.reshape(img2, (28, 28)) * 255
        print(f"img1.shape: {img1.shape}")
        label1, label2 = label.numpy()
        print(label1, label2)
        cv.imwrite(f"./imgs/img_{label1}.png", img1)
        cv.imwrite(f"./imgs/img_{label2}.png", img2)
        break


# 加载网络
mnist_net = cv.dnn.readNetFromONNX("./models/cnn_mnist.onnx")
image = cv.imread("./imgs/img_8.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(f"image.shape: {image.shape}; gray.shape: {gray.shape}")
blob = cv.dnn.blobFromImage(gray, 0.00392, (28, 28), (0,5)) / 0.5
print(f"blob.shape: {blob.shape}")
mnist_net.setInput(blob)
result = mnist_net.forward()
pred_label = np.argmax(result, 1)
print(f"predict label: {pred_label}")
# cv.imshow("input", gray)

# cv.waitKey(0)
# cv.destroyAllWindows()