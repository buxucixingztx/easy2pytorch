"""
构建卷积什么网络，完成mnist手写字符识别
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv


transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5), (0.5))])

train_ts = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)

print(f"{len(train_ts)}; {len(test_ts)}")

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
    
model = CNN_Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

for s in range(epochs):
    print(f"run in step: {s+1}")
    for i, (x_train, y_train) in enumerate(train_dl):
        y_pred = model(x_train)
        train_loss = loss_fn(y_pred, y_train)
        if (i + 1) % 100 == 0:
            print(f"{i+1}: {train_loss.item()}")
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

total = 0
correct_count = 0

model.eval()
for test_images, test_labels in test_dl:
    with torch.no_grad():
        pred_labels = model(test_images)
    predicted = torch.max(pred_labels, 1)[1]
    correct_count += (predicted == test_labels).sum()
    total += len(test_labels)

print(f"total: {total}; correct_count: {correct_count}; acc: {correct_count/total}")

torch.save(model, "./models/cnn_mnist_model.pth")