import torch
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import networkx as nx


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(34, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.linear = torch.nn.Linear(2, 4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        out = self.linear(x)
        out.relu()
        # out 为分类结果，x为最后一层输出
        return out, x


def visualize_embedding(x, color, epoch=None, loss=None):
    x = x.detach().numpy()
    plt.scatter(x[:, 0], x[:, 1], s=5, c=color)
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
    plt.show()


def visualize_graph(G, color):
    # 使用第二种方法
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), with_labels=False, node_color=color)
    plt.show()
    ## 可视化图 方法1
    # edges = pd.DataFrame()
    # edge_index = dataset.data.edge_index
    # edges['sources'] = edge_index[0, :]
    # edges['targets'] = edge_index[1, :]
    # G = nx.from_pandas_edgelist(edges, source='sources', target='targets')
    # nx.draw(G)
    # plt.show()

    # # 可视化图 方法2
    # from torch_geometric.utils import to_networkx
    #
    # G = to_networkx(dataset.data, to_undirected=True)
    # nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), with_labels=False, node_color=dataset.data.y)
    # plt.show()


if __name__ == '__main__':
    # 数据集加载
    from torch_geometric.datasets import KarateClub

    #  拳击俱乐部图：34个结点（学员），每个结点有34个特征，总共只有一个图，图中共有78条边，学员可划分成4类
    dataset = KarateClub()
    model = Net()
    print(model)
    data = dataset.data
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    maxEpoch = 200
    for epoch in range(maxEpoch):
        optimizer.zero_grad()
        out, endlinear = model(data.x, data.edge_index)
        # 半监督学习
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        if epoch==0 or epoch == maxEpoch-1:
            visualize_embedding(endlinear, data.y, epoch=epoch, loss=loss)
        loss.backward()
        optimizer.step()
    # 求准确率
    out, endlinear = model(data.x, data.edge_index)
    _, out = out.max(dim=1)
    correct = float(out[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
