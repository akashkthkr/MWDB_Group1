import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from services.Classifiers.SVM import SupportVectorMachine, gaussian_kernel

def main():

    X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.5, random_state=40)
    x_train = X[0:75]
    y_train = y[0:75]
    y_train = np.where(y_train == 0, -1, 1)
    x_test = X[75:]
    y_test = y[75:]
    y_test = np.where(y_test == 0, -1, 1)
    clf = SupportVectorMachine(gaussian_kernel, C=500)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    correct = np.sum(predictions == y_test)
    accuracy = (correct / len(predictions)) * 100
    print("Accuracy: " + str(accuracy) + "%")
if __name__ == "__main__":
    main()