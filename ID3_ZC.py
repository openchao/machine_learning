import pandas as pd
import numpy as np


class ID3():
    """
    构建混合型决策树
    """

    def __init__(self, criterion='shannon'):
        """
        初始化
        :param criterion: 使用那种计算公式计算熵，默认香浓熵
        """
        self.criterion = criterion

    def fit(self, X, y):
        """
        训练模型
        :param X:
        :param y:
        :return:
        """
        # 将X，y水平拼接并转换成DataFrame类型，方便后续统计
        self.data = np.hstack((X, y))
        self.data = pd.DataFrame(self.data)
        # 创建树
        # self.Tree 使用字典的形式存储树的结构：{'特征':{'特征':'标签'}}
        self.Tree = self.createTree(self.data)

    def createTree(self, data):
        """
        递归构建决策树
        :return:
        """
        featlist = list(data.columns)
        # 统计类别
        classlist = data.iloc[:, -1].value_counts()
        # 只有一类或者只有一列则直接返回改标签
        if classlist[0] == data.shape[0] or data.shape[1] == 1:
            return classlist.index[0]  # 若是则返回类标签
        # 确定当前分类
        axis = self.bestSplit(data)  # 确定当前最佳切分列的索引
        # 分割数据集
        bestfeat = featlist[axis]  # 获取该索引对应的特征，将其作为根节点
        myTree = {bestfeat: {}}  # 采用字典嵌套的方式存储树信息
        valuelist = set(data.iloc[:, axis])  # 提取最佳切分列的所有属性值
        for value in valuelist:  # 对每一个属性值递归建树
            # print(axis)
            myTree[bestfeat][value] = self.createTree(self.dataSetSpilt(data, axis, value))
        return myTree

    def bestSplit(self, data):
        """
        求用来划分数据集的最佳属性
        :return:
        """
        # 获得行数和列数，列数包括标签列
        rows, cols = data.shape
        cols -= 1
        bigEnt = 1  # 信息增益最大，可以转换成改列的信息熵最小，信息熵设置最大：1
        axis = -1  # 初始化最佳切分列，默认为标签列
        # 计算每一个变量的信息上
        for i in range(cols):  # data:[X,y]
            # 获得每一列的标签：data.iloc[:, i].value_counts() is Series：a:2,b:3...
            levels = data.iloc[:, i].value_counts().index
            ents = 0  # 初始化子节点的信息熵
            for j in levels:
                # data[data.iloc[:, i] == j]:取出第i列中值等于j的所有行和列
                childSet = data[data.iloc[:, i] == j]
                ent = self.entropy(childSet)  # 计算当前结点的信息熵
                ents += (childSet.shape[0] / rows) * ent  # 计算当前列的信息熵
            if ents < bigEnt:  # 选择最大信息增益
                bigEnt = ents
                axis = i
        return axis  # 返回最大信息增益所在列的索引,即信息熵最小

    def dataSetSpilt(self, data, axis, value):
        """
        按照给定的列划分数据集
        :param data: 原始数据集
        :param axis: 指定的列索引
        :param value: 指定的属性值
        :return: 按照指定列索引和属性值切分后的数据集
        """
        col = data.columns[axis]  # 指定列的索引
        SpiltDataSet = data.loc[data[col] == value, :].drop(col, axis=1)
        return SpiltDataSet

    def entropy(self, data):
        """
        计算熵
        :param data:
        :return:
        """
        # 只写了一种计算熵的方法
        if self.criterion == 'shannon':
            return self.shannon(data)

    def shannon(self, data):
        """
        香浓熵
        :param data:
        :return:
        """
        n = data.shape[0]
        iset = data.iloc[:, -1].value_counts()
        p = iset / n  # 每一类标签所占比
        # p中可能存在等于0的情况
        ent = (-p * np.log2(p)).sum()  # 计算信息熵
        return ent

    def predict(self, x):
        """
        预测
        :param x:
        :return:
        """
        x = pd.DataFrame(x)
        y_ = []
        for i in range(x.shape[0]):  # 对测试集中每一行数据(每一个实例)进行循环
            xi = x.iloc[i, :]  # 取出每行的数据部分；标签列是最后一列，根据实际dataframe确定
            classLabel = self.classify(self.Tree, xi)  # 预测该实例的分类
            y_.append(classLabel)
        return y_

    def classify(self, Tree, xi):
        """
        递归查找叶子节点
        :param Tree:
        :param xi:
        :return:
        """
        firstStr = next(iter(Tree))  # 获取决策树的根节点
        secondDict = Tree[firstStr]  # 下一个字典
        # columns = list(self.data.columns)
        # print(columns)
        # featIndex = columns.index(firstStr)  # 第一个节点对应列的索引
        classLabel = secondDict[list(secondDict.keys())[0]]  # 标签初始化
        for key in secondDict.keys():
            if xi[firstStr] == key:
                # 没有子树
                if type(secondDict[key]) == dict:
                    classLabel = self.classify(secondDict[key], xi)
                # 如果有子树递归访问直到叶子节点
                else:
                    classLabel = secondDict[key]
        return classLabel

    def save(self, filename):
        """
        保存模型
        :param filename:
        :return:
        """
        np.save(filename, self.Tree)
        print("Tree Saved in: " + filename)

    def load(self, filename):
        """
        加载模型
        :param filename:
        :return:
        """
        Tree = np.load(filename, allow_pickle=True).item()
        self.Tree = dict(Tree)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    seed = 0
    iris = datasets.load_iris()  # 加载数据集
    X = iris.data
    y = iris.target
    y = pd.DataFrame(y)
    y[y.iloc[:, 0] == 0] = 'a'
    y[y.iloc[:, 0] == 1] = 'b'
    y[y.iloc[:, 0] == 2] = 'c'
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    myTree = ID3()
    myTree.fit(X_train, y_train)
    #
    y_test_ = myTree.predict(X_test)
    y_test_ = np.asarray(y_test_).reshape((-1,))
    y_test = np.asarray(y_test).reshape((-1,))
    acc = (y_test_ == y_test).mean()
    print(acc)
    myTree.save('mytree.npy')

    # # 加载训练后的模型
    # myTree = ID3()
    # myTree.load('mytree.npy')
    # y_ = myTree.predict(X)
    # # print(y_)
    # y_ = np.asarray(y_).reshape((-1,))
    # y = np.asarray(y).reshape((-1,))
    # acc = (y == y_).mean()
    # print(acc)