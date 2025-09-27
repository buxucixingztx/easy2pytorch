"""
构建浅层神经网络
基于手写数字数据集
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

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax(dim=1)
)

model2 = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

loss_fn = nn.NLLLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fn2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

epochs = 2

for s in range(epochs):
    print(f"run in step: {s+1}")
    for i, (x_train, y_train) in enumerate(train_dl):
        x_train = x_train.view(x_train.shape[0], -1)
        y_pred = model(x_train)
        train_loss = loss_fn(y_pred, y_train)

        y_pred2 = model2(x_train)
        train_loss2 = loss_fn2(y_pred2, y_train)

        if (i+1) % 100 == 0:
            print(f"{i + 1}: train_loss1-> {train_loss.item():.3f}; | train_loss2-> {train_loss2.item():.3f}")

        model.zero_grad()
        train_loss.backward()
        optimizer.step()

        model2.zero_grad()
        train_loss2.backward()
        optimizer2.step()


total = 0
correct_count = 0
correct_count2 = 0
for test_images, test_labels in test_dl:
    for i in range(len(test_labels)):
        image = test_images[i].view(1, 784)
        with torch.no_grad():
            pred_labels = model(image)
            pred_labels2 = model2(image)

        plabels = torch.exp(pred_labels)
        probs = list(plabels.numpy()[0])
        pred_label = probs.index(max(probs))

        plabels2 = torch.exp(pred_labels2)
        probs2 = list(plabels2.numpy()[0])
        pred_label2 = probs2.index(max(probs2))

        true_label = test_labels.numpy()[i]
        if pred_label == true_label:
            correct_count += 1
        
        if pred_label2 == true_label:
            correct_count2 += 1

        total += 1

print(f"total: {total}; correct count: {correct_count}; acc: {(correct_count/total):.2f}")
print(f"total: {total}; correct count: {correct_count2}; acc: {(correct_count2/total):.2f}")

torch.save(model, "./models/nn_mnist_model.pt")