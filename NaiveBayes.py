import numpy as np
import pandas as pd


class BNC():
    def __init__(self):
        """

        """
        self.ctype = []  # 存储每列的数据类型
        self.data = []  # 存储统计数据
        self.y_class = None  # 统计后的
        self.y_len = 0

    def fit(self, X, y):
        """
        训练模型
        :param X:
        :param y:
        :return:
        """
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self.X_y = pd.concat((X, y), axis=1)
        # 统计y的种类
        self.y_class = self.X_y.iloc[:, -1].value_counts()
        self.y_len = len(self.y_class)
        # 统计每一列数据类型
        self.row, self.col = self.X_y.shape
        for i in range(self.col - 1):
            if isinstance(self.X_y.iloc[0, i], str):
                self.ctype.append(0)  # 表示分类型数据，统计个数
                temp = []
                for yc in range(self.y_len):
                    X_yc_i = self.X_y[self.X_y.iloc[:, -1] == self.y_class.index[yc]]
                    X_yc_i_count = X_yc_i.iloc[:, i].value_counts()
                    temp.append(X_yc_i_count)
                self.data.append(temp)
            else:  # 数值型数据
                self.ctype.append(1)  # 数值型数据，求均值和方差
                temp = []
                for yc in range(self.y_len):
                    X_yc_i = self.X_y[self.X_y.iloc[:, -1] == self.y_class.index[yc]]
                    X_yc_i_mean = X_yc_i.iloc[:, i].mean()
                    X_yc_i_std = X_yc_i.iloc[:, i].std()
                    temp.append([X_yc_i_mean, X_yc_i_std])
                self.data.append(temp)

    def predict(self, x):
        """
        预测数据属于哪个类别
        :return:
        """
        x = pd.DataFrame(x)
        row, col = x.shape
        if col != self.col - 1:
            print('({}, {}) cannot ({}, {})'.format(row, col, -1, self.col - 1))
            exit(-1)
        y_ = []
        for yc in range(self.y_len):
            y_likelihood = []  # 存储每行数据的p(xi|y)
            for ci, c in enumerate(self.ctype):
                if c == 0:  # 分类型数据预测
                    temp = []
                    # 统计每行数据的p(xi|y)
                    for i in range(row):
                        # 判断是否有有此属性
                        if x.iloc[i, ci] not in self.data[ci][yc].index:
                            temp.append(0)
                        else:
                            temp.append(self.data[ci][yc][x.iloc[i, ci]] / self.data[ci][yc].sum())
                    y_likelihood.append(temp)
                else:  # 数值型预测
                    y_likelihood.append(self.normal(x.iloc[:, ci], self.data[ci][yc][0], self.data[ci][yc][1]))
            # 求似然
            y_likelihood = np.asarray(y_likelihood).T
            likelihood = 1
            for ci in range(len(self.ctype)):
                likelihood *= y_likelihood[:, ci]
            likelihood *= self.y_class[self.y_class.index[yc]] / self.row
            y_.append(likelihood)
        # 计算每类的似然比
        y_ = np.asarray(y_)
        y_ = y_ / np.sum(y_)
        index = np.argmax(y_, axis=0)
        yc = []
        for i in index:
            yc.append(self.y_class.index[i])
        return yc

    def normal(self, num, mean, std):
        num = np.asarray(num)
        return 1 / np.sqrt(2 * np.pi) * np.exp(-(num - mean) ** 2 / (2 * std ** 2))


if __name__ == '__main__':
    # 获取数据
    # 萼片长度（cm）、萼片宽度（cm）、花瓣长度（cm）、花瓣宽度（cm）
    from sklearn import datasets
    from sklearn.metrics import accuracy_score

    # iris = datasets.load_iris()
    # end = -1
    # x = iris.data[0:end, :]
    # y = iris.target[0:end]
    # Model = BNC()
    # Model.fit(x, y)
    # y_ = Model.predict(x)
    # acc = accuracy_score(y, y_)
    # print(f'准确率为：{round(acc, 2)}')

    # 测试用例二
    # x = pd.DataFrame([['s', 'h', 'h', 'f'],
    #                   ['s', 'h', 'h', 'T'],
    #                   ['o', 'h', 'h', 'f'],
    #                   ['r', 'm', 'h', 'f'],
    #                   ['r', 'c', 'n', 'f'],
    #                   ['r', 'c', 'n', 't'],
    #                   ['o', 'c', 'n', 't'],
    #                   ['s', 'm', 'h', 'f'],
    #                   ['s', 'c', 'n', 'f'],
    #                   ['r', 'm', 'n', 'f'],
    #                   ['s', 'm', 'n', 't'],
    #                   ['o', 'm', 'h', 't'],
    #                   ['o', 'h', 'n', 'f'],
    #                   ['r', 'm', 'h', 't']])
    # y = pd.DataFrame(['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n', ])
    # print(x.shape)
    # print(y.shape)
    # Model = BNC()
    # Model.fit(x, y)
    # for i in range(4):
    #     print(Model.data[i])
    #     print('*'*100)
    # y_ = Model.predict(x)
    # print(y_)
    # print(['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n', ])

    # 测试用例三
    x = pd.DataFrame([['s', 'h', 1, 'f'],
                      ['s', 'h', 2, 'T'],
                      ['o', 'h', 2, 'f'],
                      ['r', 'm', 3, 'f'],
                      ['r', 'c', 5, 'f'],
                      ['r', 'c', 4, 't'],
                      ['o', 'c', 6, 't'],
                      ['s', 'm', 8, 'f'],
                      ['s', 'c', 7, 'f'],
                      ['r', 'm', 3, 'f'],
                      ['s', 'm', 6, 't'],
                      ['o', 'm', 4, 't'],
                      ['o', 'h', 2, 'f'],
                      ['r', 'm', 2, 't']])
    y = pd.DataFrame(['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n', ])
    print(x.shape)
    print(y.shape)
    Model = BNC()
    Model.fit(x, y)
    for i in range(4):
        print(Model.data[i])
        print('*'*100)
    y_ = Model.predict(x)
    print(y_)
    print(['n', 'n', 'y', 'y', 'y', 'n', 'y', 'n', 'y', 'y', 'y', 'y', 'y', 'n', ])


