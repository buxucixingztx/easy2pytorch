from torch.utils.data import DataLoader
import torchvision as tv


transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5), (0.5))])

train_ts = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)