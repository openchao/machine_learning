import h5py
import numpy as np


class BPNetwork():

    def __init__(self, model='gradient_descent', alpha=0.01, lamb=0.01, maxEpoch=1000):
        self.WList = []
        self.bList = []
        self.activationList = []
        self.aList = []
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        self.maxEpoch = maxEpoch
        self.row = 0
        self.col = 0
        self.history = {'loss': []}

    class Dense():
        def __init__(self, units, activation=None, use_bias=True,
                     kernel_initializer='gauss_uniform',
                     bias_initializer='zeros', kernel_regularizer=None,
                     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                     bias_constraint=None, **kwargs):
            """
            设置全连接层
            :param units: 正整数, 输出空间维度
            :param activation:激活函数, 若不指定, 则不适用激活函数
            :param use_bias:布尔值, 该层是否使用偏置向量
            :param kernel_initializer: kernel权值矩阵的初始化器
            :param bias_initializer:偏执向量的初始化器
            :param kernel_regularizer:运用到偏执项的正则化函数
            :param bias_regularizer:运用到偏执项的的正则化函数
            :param activity_regularizer:运用到层的输出正则化函数
            :param kernel_constraint:运用到kernel权值矩阵的约束函数
            :param bias_constraint:运用到偏执向量的约束函数
            :param kwargs:
            :return:
            """
            self.units = units
            self.activation = activation
            self.use_bias = use_bias
            self.kernel_initializer = kernel_initializer
            self.bias_initializer = bias_initializer
            self.kernel_regularizer = kernel_regularizer
            self.bias_regularizer = bias_regularizer
            self.activity_regularizer = activity_regularizer
            self.kernel_constraint = kernel_constraint
            self.bias_constraint = bias_constraint

    def Sequential(self, layers):
        """
        构建神经网络
        :param layers: list->每一层网络结构
        :return:
        """
        if len(layers) == 0:
            return
        # 判断是否使用正则化
        pass
        # y_ = np.ones((layers[-1].units, 1))
        # self.activationList.append('input')
        # 输入层只需要获取神经元个数
        WpreShape = layers[0].units
        for layer in layers[1:]:
            WnextShape = layer.units
            # 判断使用哪种函数初始化W
            if layer.kernel_initializer == 'gauss_uniform':
                W = self.gauss_uniform((WpreShape, WnextShape))
                self.WList.append(W)
            # 判断使用哪种函数初始化b
            b = np.zeros((1, WnextShape))
            if layer.bias_initializer == 'zeros':
                b = np.zeros((1, WnextShape))
            elif layer.bias_initializer == 'ones':
                b = np.ones((1, WnextShape))
            self.bList.append(b)
            WpreShape = WnextShape
            self.activationList.append(layer.activation)

    def gauss_uniform(self, size):
        """
        高斯分布
        :param size:
        :return:
        """
        return np.random.normal(size=size)

    def compile(self, optimizer='gradient_descent', loss='crossentropy', metrics=None):
        """
        声明优化器、损失函数和评价指标
        :param optimizer: 优化器
        :param loss: 损失函数
        :param metrics: 评价指标
        :return:
        """
        self.optimizer = optimizer
        self.lossFun = loss
        self.metricsList = []
        if metrics is not None:
            for metric in metrics:
                # 调用评价方法，传递方法
                self.metricsList.append(metric)

    def fit(self, X, y, batch_size=None, save=False, frequent=1):
        """
        训练模型
        :param X:
        :param y:
        :param batch_size: 每次训练多少数据
        :param epochs: 迭代次数
        :param save: 是否保存文件
        :param frequent: 保存频率
        :return:
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if len(self.y.shape) == 1:
            self.y = self.y.reshape((-1, 1))
        # 划分训练数据
        if batch_size is None:
            pass
        else:
            # 划分训练数据
            pass

        for i in range(self.maxEpoch):
            # 向前传播
            y_ = self.forward(self.X)
            print(f'{i}  {y_.shape}')
            # 反向传播
            if self.optimizer == 'gradient_descent':
                self.gradient_descent(y_)

            loss = self.crossentropy(y_)
            print(f'Epoch={i}, loss={loss}')
            self.history['loss'].append(loss)
            # 计算评价指标
            for metrics in self.metricsList:
                self.history[f'{metrics}'] = metrics
            if i%frequent == 0 and save:
                self.save(i)
        if save:
            self.save(i)

    def forward(self, X):
        """
        向前传播
        :param X:
        :return:y_
        """
        a0 = X
        self.aList = [a0]
        for activation, w, b in zip(self.activationList, self.WList, self.bList):
            h = np.dot(a0, w) + b
            if activation == 'sigmoid':
                a1 = self.sigmoid(h)
                a0 = a1
                self.aList.append(a0)
        return a0

    def gradient_descent(self, y_):
        """
        梯度下降
        :param y_:预测值
        :return:
        """
        self.row, self.col = self.X.shape
        # 最后一层求导，代码只有交叉熵和sigmoid，慢慢补
        if self.lossFun == 'crossentropy':
            deltaL = self.crossentropy(y_, deriv=True)
            # print(f'deltaL {deltaL.shape}')

        WdeltaList = []
        bdeltaList = []
        L = 0
        for w, a, activation in zip(self.WList[::-1], self.aList[1:][::-1], self.activationList[::-1]):
            # print('*' * 100)
            # print(a.T.shape, deltaL.shape)
            if activation == 'sigmoid':
                deltaL *= self.sigmoid(a, deriv=True) # deltaL是每层的残差:(该层的残差 点乘 该层的权重.T)*上层激活函数的导数
            bdeltaList.append(np.mean(deltaL, axis=0, keepdims=True)) # b的偏导：本层残差的均值
            # 隐藏层的残差 = 输出层权重.T * 输出层残差 * 本层激活函数偏导
            delta = np.dot(self.aList[::-1][L+1].T, deltaL)  # 每层权重的残差=上一层输出.T 点乘 该层残差
            L += 1
            WdeltaList.append(delta)
            deltaL = np.dot(deltaL, w.T)  # deltaL是每层的残差：该层的残差 点乘 该层的权重.T
            # print(f'deltaL {deltaL.shape}')
        # 正则化处理
        pass
        # 更新权重
        WTemp = []
        bTemp = []
        for W, Wdelta, b in zip(self.WList, WdeltaList[::-1], bdeltaList[::-1]):
            wGrad = (W-self.alpha/self.row*Wdelta).copy()
            bGrad = (b - self.alpha * b).copy()
            WTemp.append(wGrad)
            bTemp.append(bGrad)
        self.WList = WTemp
        self.bList = bTemp

    def predict(self, X):
        """
        预测数据属于哪个类别
        :return:
        """
        X = np.array(X)
        y_ = self.forward(X)
        index = np.argmax(y_, axis=1)
        return index

    def crossentropy(self, y_, deriv=False):
        """
        交叉熵函数求导
        :return:
        """
        if deriv:
            if len(self.y.shape) == 1:
                return -(self.y / y_ - (1 - self.y).reshape((-1, 1)) / (1 - y_).reshape((-1, 1)))
            else:
                return -(self.y / y_ - (1 - self.y) / (1 - y_))
        else:
            if len(self.y.shape) == 1:
                loss = -(self.y * np.log(y_) + (1 - y).reshape((-1, 1)) * np.log((1 - y_).reshape((-1, 1))))
                return np.mean(np.mean(loss))
            else:
                loss = -(self.y * np.log(y_) + (1 - self.y) * np.log(1 - y_))
                return np.mean(np.mean(loss))

    def sigmoid(self, x, deriv=False):
        """
        S型函数
        :param deriv: 是否求导
        :param x:
        :return:
        """
        if deriv:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def save(self, filename):
        """
        保存模型
        :param filename:文件路径
        :param weights:list->权重
        :return:
        """
        # 可以写一下创建目录
        # 保存权重
        with h5py.File(r'./Weights/{}.h5'.format(filename), mode='w') as f:
            i = 0
            for weight, b in zip(self.WList, self.bList):
                f['w{}'.format(i)] = weight
                f['b{}'.format(i)] = b
                i += 1

        # 保存模型结构
        with open(r'./Weights/{}.txt'.format(filename), mode='w') as f:
            for activation in self.activationList:
                f.write('{}\n'.format(activation))

    def load(self, filename):
        """
        加载模型
        :param filename:文件路径
        :return:
        """
        # 加载模型
        with open(r'{}.txt'.format(filename), mode='r') as f:
            self.activationList = f.read().split('\n')[:-1]
        layers = len(self.activationList)
        self.WList = []
        self.bList = []
        with h5py.File(r'{}.h5'.format(filename), mode='r') as f:
            for i in range(layers):
                self.WList.append(np.asarray(f['w{}'.format(i)]))
                self.bList.append(np.asarray(f['b{}'.format(i)]))





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
    x = iris.data
    y = iris.target
    BP = BPNetwork(alpha=0.015, maxEpoch=5000)
    # 模型一
    # BP.Sequential([
    #     BP.Dense(4),
    #     BP.Dense(64, activation='sigmoid'),
    #     BP.Dense(128, activation='sigmoid'),
    #     BP.Dense(64, activation='sigmoid'),
    #     BP.Dense(3, activation='sigmoid')
    # ])

    # 模型二
    BP.Sequential([
        BP.Dense(4),
        BP.Dense(5, activation='sigmoid'),
        BP.Dense(3, activation='sigmoid')
    ])

    # 使用oen hot编码
    y = np.asarray(y)
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, 3))
    one_hot[np.arange(n_samples), y.T] = 1
    # # 训练模型
    # for W in BP.WList:
    #     print(W.shape)
    # BP.compile(optimizer='gradient_descent', loss='crossentropy')
    # BP.fit(x, one_hot, save=False, frequent=50)
    # y_ = BP.predict(x)
    # acc = accuracy_score(y, y_)
    # print(f'准确率为：{round(acc, 2)}')
    # loss = BP.history['loss']
    # x = range(len(loss))
    # plt.plot(x, loss)
    # plt.xlabel('Epochs')
    # plt.ylabel('loss')
    # plt.show()

    # 加载模型
    BPLoad = BPNetwork()
    BPLoad.load('./Weights/4999')
    for W in BPLoad.WList:
        print(W.shape)
    y_ = BPLoad.predict(x)
    acc = accuracy_score(y, y_)
    print(f'准确率为：{round(acc, 2)}')