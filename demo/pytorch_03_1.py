"""
将模型转换为onnx格式
"""
import torch
from torch import nn


class CNN_Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*32, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 7*7*32)
        x = self.fc_layers(x)
        return x
    

dummy_input = torch.randn(1, 1, 28, 28)
# model = torch.load("./models/cnn_mnist_model.pt")
model = torch.load("./models/cnn_mnist_model.pth", weights_only=False)
torch.onnx.export(model, 
                  dummy_input, 
                  "./models/cnn_mnist.onnx", 
                  verbose=True, 
                  input_names=['input'],
                  output_names=['out'])

