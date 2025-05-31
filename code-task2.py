import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import vgg11, vgg11_bn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 4000

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 准备模型
def get_model(use_bn):
    model = vgg11_bn(pretrained=False) if use_bn else vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)
    return model.to(device)

# 训练模型并记录 loss
def train(model, lr, steps=STEPS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    losses = []
    data_iter = iter(trainloader)
    
    for step in tqdm(range(steps)):
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(trainloader)
            inputs, labels = next(data_iter)

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses, model

# 实验主流程
def run_experiment(use_bn):
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    all_loss_lists = []

    for lr in learning_rates:
        model_type = "vgg_bn" if use_bn else "vgg"
        print(f"Training {model_type.upper()} with lr={lr}")
        model = get_model(use_bn)
        losses, trained_model = train(model, lr)
        all_loss_lists.append(losses)

        # 保存模型
        model_save_path = f"{model_type}_lr{lr:.0e}.pth"
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    all_losses = np.array([np.pad(l, (0, max(map(len, all_loss_lists)) - len(l)), 'edge') for l in all_loss_lists])
    min_curve = np.min(all_losses, axis=0)
    max_curve = np.max(all_losses, axis=0)
    return min_curve, max_curve

# 运行实验
min_curve_nobn, max_curve_nobn = run_experiment(use_bn=False)
min_curve_bn, max_curve_bn = run_experiment(use_bn=True)

# 可视化
plt.figure(figsize=(10, 6))
x = np.arange(len(min_curve_nobn))

plt.fill_between(x, min_curve_nobn, max_curve_nobn, alpha=0.4, color='green', label='Standard VGG')
plt.plot(x, min_curve_nobn, 'g--', linewidth=0.8)
plt.plot(x, max_curve_nobn, 'g--', linewidth=0.8)

plt.fill_between(x, min_curve_bn, max_curve_bn, alpha=0.4, color='red', label='Standard VGG + BatchNorm')
plt.plot(x, min_curve_bn, 'r-', linewidth=0.8)
plt.plot(x, max_curve_bn, 'r-', linewidth=0.8)

plt.title("Loss Landscape")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./loss.pdf")