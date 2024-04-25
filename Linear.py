import numpy as np


class Linear():
    def __init__(self, model='gradient_descent', alpha=0.01, maxEpoch=1000, E=0.01):
        self.model = model
        self.alpha = alpha
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
        self.col += 1
        # 在源数据前面添加数值全为1的一列
        X = np.column_stack((np.ones((self.row, 1)), self.X))
        # 初始化权重
        Wper = np.random.normal(size=(self.col, 1))  # 记录更新前的权重
        self.W = Wper - self.alpha / self.row * np.dot(X.T, (np.dot(X, Wper) - self.y))
        count = 0
        while abs(np.mean(abs(self.W - Wper)) - self.E) >= 0 and count < self.maxEpoch:
            Wper = np.copy(self.W)
            self.W = Wper - self.alpha / self.row * np.dot(X.T, (np.dot(X, Wper) - self.y))
            count += 1
            y_ = np.dot(X, self.W)
            loss = self.loss(self.y, y_)
            print(f'Epoch={count}, loss={loss}')
            self.history['loss'].append(loss)
        self.coef_ = self.W[1:, 0]
        self.intercept_ = self.W[0, 0]

    def predict(self, x):
        """
        预测数据属于哪个类别
        :return:
        """
        x = np.array(x)
        row, col = x.shape
        if col != self.col - 1:
            print('({}, {}) cannot ({}, {})'.format(row, col, -1, self.col - 1))
            exit(-1)
        x = np.column_stack((np.ones((row, 1)), x))
        y_ = np.dot(x, self.W)
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

    def loss(self, y, y_):
        """
        求RMSE
        :param y:
        :param y_:
        :return:
        """
        print(np.mean((y-y_)**2))
        return np.mean((y-y_)**2)


if __name__ == '__main__':
    # 获取数据
    from sklearn import datasets
    from sklearn.metrics import accuracy_score

    from matplotlib import pyplot as plt
    import matplotlib

    font = {'family': 'MicroSoft YaHei',
            'weight': 'bold',
            'size': '7'}
    matplotlib.rc("font", **font)

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    Model = Linear()
    Model.fit(x, y)
    Model.getEquation()
    y_ = Model.predict(x)
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

