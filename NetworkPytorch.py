import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data


class Net(nn.Module):
    """
    构建网络模型
    input_size = 28 * 28
    hidden_size = 500:ReLU
    num_classes = 10:Softmax(dim=1)
    loss CrossEntropyLoss
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

    def softmax(self, x):
        m = nn.Softmax(dim=1)
        return m(x)


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":
    batch_size = 64
    learning_rate = 0.001
    # 加载数据
    train_dataset = torchvision.datasets.MNIST(
        root=r'D:\深度学习数据', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(
        root=r'D:\深度学习数据', train=False, download=True, transform=torchvision.transforms.ToTensor())
    # 导入数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    net = Net()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))  # 声明变量用于学习参数。view相当于reshape
            labels = Variable(labels)
            optimizer.zero_grad()  # 梯度清零
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()  # 前向传播
            optimizer.step()  # 反向传播
            # 打印指标
            if (i + 1) % 100 == 0:
                print('Epoch: [{}/{} ] Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                    loss.item()
                ))
    # 测试集验证准确率
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = Variable(images.view(-1, 28 * 28))  # 声明变量用于学习参数。view相当于reshape
        outputs = net(images)
        _, predicted = torch.max(outputs.data, dim=1)  # value, index
        total += labels.size(0)  # 多少数据
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images:{:.4f}%'.format(100*correct/total))