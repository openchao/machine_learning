import numpy as np


class Logistic():
    def __init__(self, model='gradient_descent', alpha=0.01, lamb=0.01, maxEpoch=1000, E=0.01):
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        self.maxEpoch = maxEpoch
        self.E = E
        self.coef_ = []
        self.intercept_ = 0
        self.row = 0
        self.col = 0
        self.history = {'loss': []}

    def fit(self, X, y):
        """
        训练模型
        :param X:
        :param y:
        :return:
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape((-1, 1))
        if self.model == 'gradient_descent':
            self.gradient_descent()
        elif self.model == 'least_squares':
            self.least_squares()

    def least_squares(self):
        """
        最小二乘法
        :return:
        """

    def gradient_descent(self):
        """
        梯度下降
        :return:
        """
        self.row, self.col = self.X.shape
        # 初始化权重
        Wper = np.random.normal(size=(self.col, 1))  # 记录更新前的权重
        Bper = np.random.normal()

        z = np.dot(self.X, Wper) + Bper
        y_y = self.sigmoid(z) - self.y
        self.b = Bper - self.alpha / np.sum(self.row * (y_y))
        self.W = Wper - self.alpha / self.row * np.dot(self.X.T, y_y) - self.lamb * Wper
        count = 0
        while abs(np.mean(abs(self.W - Wper)) - self.E) >= 0 and count < self.maxEpoch:
            Wper = np.copy(self.W)
            Bper = np.copy(self.b)
            z = np.dot(self.X, Wper) + Bper
            y_y = self.sigmoid(z) - self.y
            self.b = Bper - self.alpha / np.sum(self.row * (y_y))
            self.W = Wper - self.alpha / self.row * np.dot(self.X.T, y_y) - self.lamb * Wper

            count += 1
            y_ = np.dot(self.X, self.W) + self.b
            loss = self.loss(self.y, y_)
            print(f'Epoch={count}, loss={loss}')
            self.history['loss'].append(loss)
        self.coef_ = self.W[:, 0]
        self.intercept_ = self.b

    def predict(self, x):
        """
        预测数据属于哪个类别
        :return:
        """
        x = np.array(x)
        row, col = x.shape
        if col != self.col:
            print('({}, {}) cannot ({}, {})'.format(row, col, -1, self.col - 1))
            exit(-1)
        y_ = np.dot(x, self.W) + self.b
        return y_

    def getEquation(self):
        """
        获取多元线性回归方程
        :return: string
        """
        s = 'f='

        for i, elem in enumerate(self.coef_):
            s += '{} * x{} + '.format(round(elem, 4), i)
        s += '{}'.format(round(self.intercept_, 4))
        return s

    def sigmoid(self, x):
        """
        S型函数
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def loss(self, y, y_):
        """
        求RMSE
        :param y:
        :param y_:
        :return:
        """
        print(np.mean((y - y_) ** 2))
        return np.mean((y - y_) ** 2)


if __name__ == '__main__':
    # 获取数据
    # 萼片长度（cm）、萼片宽度（cm）、花瓣长度（cm）、花瓣宽度（cm）
    from sklearn import datasets
    from sklearn.metrics import accuracy_score

    from matplotlib import pyplot as plt
    import matplotlib

    font = {'family': 'MicroSoft YaHei',
            'weight': 'bold',
            'size': '7'}
    matplotlib.rc("font", **font)

    iris = datasets.load_iris()
    end = 100
    x = iris.data[0:end, :]
    y = iris.target[0:end]
    Model = Logistic()
    Model.fit(x, y)
    Model.getEquation()
    y_ = Model.predict(x)
    y_[np.where(y_ > 0.5)] = 1
    y_[np.where(y_ <= 0.5)] = 0
    y_ = np.around(y_)  # 四舍五入
    # 显示loss
    acc = accuracy_score(y, y_)
    print(f'准确率为：{round(acc, 2)}')

    loss = Model.history['loss']
    x = range(len(loss))
    plt.plot(x, loss)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()
