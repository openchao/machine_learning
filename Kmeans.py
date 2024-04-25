import numpy as np
from sklearn import datasets
from sklearn.svm import LinearSVC


class Kmeans():

    def __init__(self, n_clusters=3, dist='abs', E=0.0001):
        self.n_clusters = n_clusters
        self.dist = dist
        self.E = E
        self.kx = []  # 记录每个类别在源数据的下标

    def fit(self, data):
        """
        训练模型
        :param data:
        :return:
        """
        # 临近均值E
        # 获得行数和列数
        self.data = np.asarray(data)
        (row, col) = data.shape
        # 随机分配中心点
        randList = []
        np.random.seed(0)
        for randi in range(self.n_clusters):
            rand = np.random.randint(0, row)
            while rand in randList:
                rand = np.random.randint(0, row)
            randList.append(rand)
        self.clusters = data[randList, :]

        # self.clusters = np.asarray([data[0,:], data[55,:], data[110,:]])
        # 初始化距离
        d = np.zeros((self.n_clusters, row, 1))
        while 1:
            for i in range(self.n_clusters):
                # 求距离
                d[i] = self.distance(data, self.clusters[i])
            # 返回最小值下标
            mind = np.argmin(d, axis=0)
            mind = np.array(mind[:, 0])
            # 把点分类到属于自己的类别
            for i in range(self.n_clusters):
                self.kx.append(np.where(mind == i)[0])
            # 计算中心点和误差
            e = np.zeros((2, self.n_clusters, 1))  # 计算上一次与当前这一次的距离差别多大
            for i in range(self.n_clusters):
                # 获取第i个对象的行数
                rowx = self.kx[i].shape[0]
                data_x1 = np.zeros((rowx, col))
                data_x2 = np.zeros((self.n_clusters, rowx, col))
                for j in range(rowx):
                    # 每个对象和第i个中心点作差
                    data_x1[j] = self.distance(data[self.kx[i][j]].reshape((1,-1)), self.clusters[i])
                    data_x2[i, j] = data[self.kx[i][j]]
                # 求方差
                e[0, i, 0] = np.sum(np.sum(data_x1 ** 2))
                # 更新第i类的中心点
                self.clusters[i] = np.mean(data_x2[i], axis=0)
                for j in range(rowx):
                    # 每个对象和第i个中心点作差
                    data_x1[j] = self.distance(data[self.kx[i][j]].reshape((1,-1)), self.clusters[i])
                e[1, i, 0] = np.sum(np.sum(data_x1 ** 2))
            # 检测聚类是否满足要求
            i = 0
            while i < self.n_clusters:
                if abs(e[0, i, 0] - e[1, i, 0]) <= self.E:
                    break
                i += 1
            if i == self.n_clusters:
                break

    def distance(self, x1, x2):
        """
        计算两点之间的距离
        :param x1:
        :param x2:
        :return: 距离矩阵
        """
        if self.dist == 'abs':
            dev = abs(x1 - x2)
            return np.sum(dev, axis=1, keepdims=True)
        if self.dist == 'square':
            dev = (x1 - x2) ** 2
            return np.sqrt(np.sum(dev, axis=1, keepdims=True))

    def predict(self, x):
        """
        预测数据属于哪个类别
        :return:
        """
        x = np.array(x).reshape((1,-1))
        d = []
        for i in range(self.n_clusters):
            # 求距离
            d.append(self.distance(x, self.clusters[i]))
        # 返回最小值下标
        return np.argmin(d)

    def getClusters(self):
        """

        :return: 返回类别中心的下标和类别中心
        """
        return self.kx, self.clusters

    def GiNi(self):
        """
        使用DB指数评价聚类好坏
        :return:
        """

    def Silhouette(self):
        """
        使用轮廓系数评价聚类好坏
        :return:
        """


# 获取数据
# 萼片长度（cm）、萼片宽度（cm）、花瓣长度（cm）、花瓣宽度（cm）
iris = datasets.load_iris()
x = iris.data
y = iris.target

Model = Kmeans(n_clusters=3, dist='abs')
Model.fit(data=x)
index, clusters = Model.getClusters()
# 打印
for i in range(3):
    print(index[i].shape)
    print(index[i])
    print(clusters[i])

print(Model.predict(x[0,:]))
