import copy


class DecisionTree(object):

    def __init__(self):
        self.max_depth = 20
        self.min_size = 5
        self.tree = {}

    # Fit training data
    def fit(self, x_train, y_train):
        train_set = copy.deepcopy(x_train)
        for i in range(len(train_set)):
            train_set[i].append(y_train[i])
        self.tree = self.decision_tree(train_set)
        # return self

    # Split a dataset based on an attribute and an attribute value
    def dataset_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def calc_gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def get_best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.dataset_split(index, row[index], dataset)
                gini = self.calc_gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a leaf node value
    def leaf_node(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.leaf_node(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.leaf_node(left), self.leaf_node(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.get_best_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.leaf_node(right)
        else:
            node['right'] = self.get_best_split(right)
            self.split(node['right'], depth+1)

    # Make a prediction with a decision tree
    def predict1(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict1(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict1(node['right'], row)
            else:
                return node['right']

    # Classification and Regression Tree Algorithm
    def decision_tree(self, train):
        root = self.get_best_split(train)
        self.split(root, 1)
        return root

    def predict(self, test):
        predictions = list()
        for row in test:
            # print("row:::: {}".format(row))
            # print("node::: {}".format(self.tree))
            prediction = self.predict1(self.tree, row)
            # print("prediction: {}".format(prediction))
            predictions.append(prediction)
        return predictions
