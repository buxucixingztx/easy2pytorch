"""
构建循环神经网络
"""
import torch
from torch import nn
import numpy as np
from utils import *


# 经典RNN网络
class RNN_Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn_layers = nn.RNN(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.out_layers = nn.Linear(64, 10)

    def forward(self, x):
        r_out, hn = self.rnn_layers(x, None)
        out = self.out_layers(r_out[:, -1, :])
        return out
    

# LSTM网络
class LSTM_Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn_layers = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.out_layer = nn.Linear(64, 10)

    def forward(self, x):
        r_out, hn = self.rnn_layers(x, None)
        out = self.out_layer(r_out[:, -1, :])
        return out
    

rnn_model = RNN_Net()
# rnn_model = LSTM_Net()
print(rnn_model)
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

epochs = 2

for epoch in range(epochs):
    print(f"run epoch-{epoch+1}:")
    for i, (x_train, y_train) in enumerate(train_dl):
        x_train = x_train.view(-1, 28, 28)
        y_pred = rnn_model(x_train)
        train_loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f"{i+1}: {train_loss.item()}")


total = 0
correct_count = 0
for test_images, test_labels in test_dl:
    for i in range(len(test_labels)):
        image = test_images[i].view(1, 28, 28)
        with torch.no_grad():
            pred_labels = rnn_model(image)
        # probs = list(pred_labels.numpy()[0])
        pred_label = np.argmax(pred_labels.numpy())
        true_label = test_labels.numpy()[i]
        if pred_label == true_label:
            correct_count += 1

        total += 1

print(f"total: {total}; correct_count: {correct_count}; acc: {correct_count/total}")

torch.save(rnn_model, "./models/rnn_mnist_model.pth")
# torch.save(rnn_model, "./models/lstm_mnist_model.pth")