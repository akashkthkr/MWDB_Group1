
import copy

from random import randrange


class DecisionTreeClassifier:

    def __init__(self):
        self.tree = None
        pass

    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def gini_index(self, groups, classes):
        # count all samples at split point
        # print("groups length = {}".format(groups))
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        # print("class values: {}".format(class_values))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                # print("rowwww : {}".format(row))
                groups = self.test_split(index, row[index], dataset)
                # print("index::: {}  value: {} --- groups: {}".format(index, row[index], groups))
                gini = self.gini_index(groups, class_values)
                # print("gini == {} ,, b_score == {}".format(gini, b_score))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    def build_tree(self, train, max_depth, min_size):
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    def predictTest(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predictTest(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predictTest(node['right'], row)
            else:
                return node['right']

    def fit(self, x, y):
        train = copy.deepcopy(x)
        labels = copy.deepcopy(y)
        # print("x len: {}  x:::: {}".format(len(x), x))
        # print("y len: {}  y:::: {}".format(len(y), y))
        for i in range(0, len(train)):
            train[i].append(labels[i])
        self.tree = self.build_tree(train, float("inf"), 1)

    def predict(self, test):
        predictions = list()
        # print("Treeee: {}".format(self.tree))
        # print("row length: {}".format(len(test)))
        for row in test:
            prediction = self.predictTest(self.tree, row)
            predictions.append(prediction)
        return predictions

    def decision_tree(self, train, labels, test, max_depth, min_size):
        for i in range(0,len(train)):
            row = train[i].extend(labels[i])

        tree = self.build_tree(train, max_depth, min_size)
        predictions = list()
        for row in test:
            prediction = self.predict(tree, row)
            predictions.append(prediction)
        return predictions
