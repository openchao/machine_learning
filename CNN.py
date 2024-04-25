import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data


class Inception(nn.Module):
    """
    构建googlenet中的Inception块
    """

    def __init__(self, n_input, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1):
        """

        :param n_input: 输入的channels
        :param n1_1: ptah1的1x1Conv的channels
        :param n2_1:ptah2的1x1Conv的channels
        :param n2_3:ptah2的3x3Conv的channels
        :param n3_1:ptah3的1x1Conv的channels
        :param n3_5:ptah3的5x5Conv的channels
        :param n4_1:ptah4的1x1Conv的channels
        """
        super().__init__()
        self.relu = nn.ReLU()
        # path1
        self.p1_conv_1 = nn.Conv2d(n_input, n1_1, kernel_size=(1, 1))
        # path2
        self.p2_conv_1 = nn.Conv2d(n_input, n2_1, kernel_size=(1, 1))
        self.p2_conv_3 = nn.Conv2d(n2_1, n2_3, kernel_size=(3, 3), padding=1)
        # path3
        self.p3_conv_1 = nn.Conv2d(n_input, n3_1, kernel_size=(1, 1))
        self.p3_conv_5 = nn.Conv2d(n3_1, n3_5, kernel_size=(5, 5), padding=2)
        # path3
        self.p4_pool_3 = nn.MaxPool2d(3, padding=1, stride=1)
        self.p4_conv_1 = nn.Conv2d(n_input, n4_1, kernel_size=(5, 5), padding=2)

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p1 = self.relu(p1)

        p2 = self.relu(self.p2_conv_1(x))
        p2 = self.relu(self.p2_conv_3(p2))

        p3 = self.relu(self.p3_conv_1(x))
        p3 = self.relu(self.p3_conv_5(p3))

        p4 = self.p4_conv_1(self.p4_pool_3(x))
        p4 = self.relu(p4)

        return torch.cat([p1, p2, p3, p4], dim=1)


class Net(nn.Module):
    """
    构建网络模型预测
    这里只使用一个block
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        # block1
        block1 = nn.Sequential(
            Inception(1, 64, 96, 128, 16, 32, 32),
            # nn.Conv2d(256, 1, kernel_size=(5, 5)),
            nn.Flatten(),
            nn.Linear(200704, 10)
        )
        self.net = nn.Sequential()
        self.net.add_module('block1', block1)

    def forward(self, x):
        out = x
        for i, block in enumerate(self.net):
            out = block(out)
            if self.verbose:
                print(f'Block {i+1} output {out.size()}')
        return out

if __name__ == "__main__":
    # 测试网络结构
    # net = Net(verbose=True)
    # x = torch.normal(0, 1, size=(32, 1, 64, 64))
    # y = net(x)
    # for name, parameters in incp.named_parameters():
    #     print(name, ':', parameters.size())
    batch_size = 64
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

    # 训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.cuda(0)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    learning_rate = 0.002
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).to(device)  # 声明变量用于学习参数。view相当于reshape
            labels = Variable(labels).to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()  # 前向传播
            optimizer.step()  # 反向传播
            # 打印指标
            if (i + 1) % 50 == 0:
                print('Epoch: [{}/{} ] Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                    loss.item()
                ))


    # 测试集验证准确率
    correct = 0
    total = 0
    for images, labels in train_loader:
        # labels.to(device)
        labels = labels.to(device)
        images = Variable(images).to(device)  # 声明变量用于学习参数。view相当于reshape
        outputs = net(images)
        _, predicted = torch.max(outputs.data, dim=1)  # value, index
        total += labels.size(0)  # 多少数据
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images:{:.4f}%'.format(100 * correct / total))

