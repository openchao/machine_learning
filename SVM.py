from sklearn.svm import SVR, SVC


class SVM():
    def __init__(self, model='SVC', param={}):
        self.model = model
        if self.model == 'SVC':
            self.Model = SVC(**param)
        elif self.model == 'SVR':
            self.Model = SVR(**param)
        self.best_parameters = param

    def fit(self, X, y):
        """
        训练模型
        :param X:
        :param y:
        :return:
        """
        self.X = X
        self.y = y
        print(X, y)
        self.Model.fit(self.X, self.y)

    def optimize(self, param_grid, scoring='accuracy', n_jobs=-1, cv=10, verbose=1):
        """
        优化模型参数
        :param verbose: verbose:日志冗长度。默认为0：不输出训练过程；1：偶尔输出；>1：对每个子模型都输出。
        :param cv: 交叉验证的次数
        :param n_jobs: n_jobs:并行数，int类型。(-1：跟CPU核数一致；1:默认值)
        :param scoring: 评价指标
        :param param_grid: 要调参数的列表(带有参数名称作为键的字典)
        :return:
        """
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.Model, param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, scoring=scoring, cv=cv)
        grid_search.fit(self.X, self.y)
        self.best_parameters = grid_search.best_estimator_.get_params()  # 获取最佳模型中的最佳参数
        # grid_search.best_params_与grid_search.best_estimator_.get_params()两者区别在于前者只返回用户搜索的参数
        # grid_search.cv_results_:给出不同参数情况下的评价结果。
        # grid_search.best_params_:已取得最佳结果的参数的组合
        print(f"best parameters are {self.best_parameters}")
        # grid_search.best_score_:优化过程期间观察到的最好的评分。
        print(f"best score are {grid_search.best_score_}")

        # 使用优化后的模型进行训练数据
        if self.model == 'SVC':
            self.Model = SVC(**self.best_parameters)
        elif self.model == 'SVR':
            self.Model = SVR(**self.best_parameters)
        self.Model.fit(self.X, self.y)

    def predict(self, X):
        """
        预测
        :param X:
        :return:
        """
        return self.Model.predict(X)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # std = StandardScaler()
    # x = std.fit_transform(x)
    # 划分数据集
    seed = 1657
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
    model = SVM(model='SVC')
    model.fit(X_train, y_train)
    # 优化模型
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.01]}
    model.optimize(param_grid=param_grid)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    train_score = []
    test_score = []
    for metric in metrics:
        train_score.append(round(metric(y_train, y_train_pred), 2))
        test_score.append(round(metric(y_test, y_test_pred), 2))
    print('回归方差', '平均绝对误差', '均方差', 'R^2')
    print('训练集:', train_score)
    print('测试集:', test_score)
