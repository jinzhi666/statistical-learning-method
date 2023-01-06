import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

# 导入数据
def create_data():
    iris = load_iris()
    # 创建 df
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 标签
    df['label'] = iris.target
    # 列数据
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]

    data = np.array(df.iloc[:, :])

    return data[:, :-1], data[:, -1]


X, y = create_data()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("部分训练集数据如下")
print(X_train[0:25])
print(y_train[0:25])

print("部分测试集数据如下")
print(X_test[0:25])
print(y_test[0:25])


class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        label = sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label

    # 计算分数
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
            else:
                print("错误有")
                print("真实值为%.0f, 预测值为%.0f\n" % (y, label))
        return right / float(len(X_test))


# 导入模型
model = NaiveBayes()

# 训练数据
model.fit(X_train, y_train)

# 输出分数
print("分数为", model.score(X_test, y_test))
