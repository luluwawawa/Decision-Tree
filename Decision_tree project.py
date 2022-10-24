# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Project 1: Decision Tree
# Goal: Implement Hung's algorithm for decision tree classification
# Data: Adult dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Name: Qinlu HU
# SID: 1155168530
# Date: 2022/10/23
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from math import log
# create decision tree object
class DecisionTree:
    train_data = []
    train_label = []
    feature_values = {}
    data = []

    def __init__(self, train_data, train_label, test_data, threshold):
        self.Read_data(train_data, train_label)
        self.threshold = threshold
        self.tree = self.Create_decision_tree(range(0, len(train_label)), range(0, len(train_data[0])))
        self.prediction = self.Classify_test_data(test_data)

    def Read_data(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label
        for data in train_data:
            for index, value in enumerate(data):
                if not index in self.feature_values.keys():
                    self.feature_values[index] = [value]
                if not value in self.feature_values[index]:
                    self.feature_values[index].append(value)

    def lable_number(self, dataset):
        lable_number = {}
        for i in dataset:
            if train_label[i] in lable_number.keys():
                lable_number[train_label[i]] += 1
            else:
                lable_number[train_label[i]] = 1

        return lable_number

    def Divide_data(self, dataset, feature, value):
        reslut = []
        for index in dataset:
            if self.train_data[index][feature] == value:
                reslut.append(index)
        return reslut

    def Input(self, dataset):
        lable_number = self.lable_number(dataset)
        size = len(dataset)
        result = 0
        for i in lable_number.values():
            pi = i / float(size)
            result -= pi * (log(pi) / log(2))
        return result

    def get(self, dataset, feature):
        values = self.feature_values[feature]
        result = 0.0
        for v in values:
            sub_data = self.Divide_data(dataset=dataset, feature=feature, value=v)
            result += len(sub_data) / float(len(dataset)) * self.Input(sub_data)
        return self.Input(dataset=dataset) - result

    def Create_decision_tree(self, dataset, features):
        lable_number = self.lable_number(dataset)
        if not features:
            return max(list(lable_number.items()), key=lambda x: x[1])[0]
        if len(lable_number) == 1:
            return list(lable_number.keys())[0]
        l = list(map(lambda x: [x, self.get(dataset=dataset, feature=x)], features))
        if len(l) == 0:
            return max(list(lable_number.items()), key=lambda x: x[1])[0]
        feature, gain = max(l, key=lambda x: x[1])
        if self.threshold > gain:
            return max(list(lable_number.items()), key=lambda x: x[1])[0]
        tree = {}
        sub_features = filter(lambda x: x != feature, features)
        tree['feature'] = feature
        for value in self.feature_values[feature]:
            sub_data = self.Divide_data(dataset=dataset, feature=feature, value=value)
            if not sub_data:
                continue
            tree[value] = self.Create_decision_tree(dataset=sub_data, features=sub_features)
        return tree

    def Classify_data(self, data):
        def f(tree, data):
            if type(tree) != dict:
                return tree
            else:
                return f(tree[data[tree['feature']]], data)

        return f(self.tree, data)

    def Classify_test_data(self, test_data):
        predict = []
        for i in range(len(test_data)):
            predict.append(self.Classify_data(test_data[i]))
        return predict

# read data
def Read_data(train_data, feature_values):
    for data in train_data:
        for index, value in enumerate(data):
            if not index in feature_values.keys():
                feature_values[index] = []
                feature_values[index].append(value)
            if not value in feature_values[index]:
                feature_values[index].append(value)
    return feature_values

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    print("Start")
    print("--------------------------------")
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    feature_values = {}

    # pre cleaning data
    for line in open('adult.data'):
        new_line = line.split(',')
        if not ' ?' in new_line and len(new_line) == 15:
            del new_line[-2]
            del new_line[2]
            train_data.append(new_line[:-1])
            if '<=50K' in new_line[-1]:
                train_label.append(0)
            elif '>50K' in new_line[-1]:
                train_label.append(1)
            else:
                train_label.append(2)

    test = []
    for line in open('adult.test'):
        new_line = line.split(',')
        if not ' ?' in new_line and len(new_line) == 15:
            del new_line[-2]
            del new_line[2]
            test_data.append(new_line[:-1])
            test.append(new_line[:-1])
            if '<=50K' in new_line[-1]:
                test_label.append(0)
            elif '>50K' in new_line[-1]:
                test_label.append(1)
            else:
                test_label.append(2)

    feature_values = Read_data(train_data, feature_values)

    max9 = max10 = max11 = 0
    for i in range(len(train_data)):
        if int(train_data[i][9]) > max9:
            max9 = int(train_data[i][9])
        if int(train_data[i][10]) > max10:
            max10 = int(train_data[i][10])
        if int(train_data[i][11]) > max11:
            max11 = int(train_data[i][11])

    for i in range(len(train_data)):
        for j in range(12):
            if len(feature_values[j]) < 20 and j != 3:
                train_data[i][j] = feature_values[j].index(train_data[i][j])
            elif j == 0:
                train_data[i][j] = int(train_data[i][j]) // 10 - 1
            elif j == 3:
                if int(train_data[i][j]) < 5:
                    train_data[i][j] = 0
                elif int(train_data[i][j]) >= 5 and int(train_data[i][j]) <= 10:
                    train_data[i][j] = 1
                else:
                    train_data[i][j] = 2
            elif j == 9:
                train_data[i][j] = int(12 * int(train_data[i][j]) / max9)
                if train_data[i][j] == 12:
                    train_data[i][j] -= 1
            elif j == 10:
                train_data[i][j] = int(12 * int(train_data[i][j]) / max10)
                if train_data[i][j] == 12:
                    train_data[i][j] -= 1
            else:
                train_data[i][j] = int(12 * int(train_data[i][j]) / max11)
                if train_data[i][j] == 12:
                    train_data[i][j] -= 1

    for i in range(len(test_data)):
        for j in range(12):
            if len(feature_values[j]) < 20 and j != 3:
                test_data[i][j] = feature_values[j].index(test_data[i][j])
            elif j == 0:
                test_data[i][j] = int(test_data[i][j]) // 10 - 1
            elif j == 3:
                if int(test_data[i][j]) < 5:
                    test_data[i][j] = 0
                elif int(test_data[i][j]) >= 5 and int(test_data[i][j]) <= 10:
                    test_data[i][j] = 1
                else:
                    test_data[i][j] = 2
            elif j == 9:
                test_data[i][j] = int(12 * int(test_data[i][j]) / max9)
                if test_data[i][j] == 12:
                    test_data[i][j] -= 1
            elif j == 10:
                test_data[i][j] = int(12 * int(test_data[i][j]) / max10)
                if test_data[i][j] == 12:
                    test_data[i][j] -= 1
            else:
                test_data[i][j] = int(12 * int(test_data[i][j]) / max11)
                if test_data[i][j] == 12:
                    test_data[i][j] -= 1

    feature_values = {}
    feature_values = Read_data(train_data, feature_values)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # create decision tree
    tree = DecisionTree(train_data=train_data, train_label=train_label, test_data=test_data, threshold=0)
    print("Decision tree is:", tree.tree)  # print tree
    prediction = tree.prediction
    print("len(test data is)：",len(test_data))
    print("len(prediction is)：",len(prediction))
    print("test data prediction is: ",prediction)
    err = 0
    for i in range(len(test_label)):
        if (prediction[i] != test_label[i]):
            err = err + 1
    acc = 1 - err / len(test_label)
    print("Accuracy rate is:", acc)  # print accuracy rate
    print("--------------------------------")
    print("End")