import cvxopt as optimizer
import cvxopt.solvers as solver
import numpy as np
from numpy import linalg



def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SupportVectorMachine(object):

    # initializing values
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        number_samples, number_features = X.shape

        # Gram matrix
        # initializing matrix of zeros and size of training data
        K = np.zeros((number_samples, number_samples))

        # getting polynomial kernel for each sample and storing in K
        for i in range(number_samples):
            for j in range(number_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # G and A are sparse matrices
        # P is a square dense or sparse real matrix, which represents a positive semidefinite symmetric matrix
        # q is a real single-column dense matrix
        # h and b are real-single column dense matrices
        # G and A are real dense or sparse matrices

        P = optimizer.matrix(np.outer(y, y) * K)
        q = optimizer.matrix(np.ones(number_samples) * -1)
        A = optimizer.matrix(y, (1, number_samples), 'd')
        b = optimizer.matrix(0.0)

        if self.C is None:
            G = optimizer.matrix(np.diag(np.ones(number_samples) * -1))
            h = optimizer.matrix(np.zeros(number_samples))
        else:
            tmp1 = np.diag(np.ones(number_samples) * -1)
            tmp2 = np.identity(number_samples)
            G = optimizer.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(number_samples)
            tmp2 = np.ones(number_samples) * self.C
            h = optimizer.matrix(np.hstack((tmp1, tmp2)))

        # solves quadratic programming problem
        solution = solver.qp(P, q, G, h, A, b)

        # calculates Lagrange multipliers
        a = np.ravel(solution['x'])

        # support vectors have non zero lagrange multipliers
        support_vectors = a > 1e-5
        ind = np.arange(len(a))[support_vectors]
        self.a = a[support_vectors]
        self.support_vectors = X[support_vectors]
        self.support_vectors_y = y[support_vectors]
        # print("---------------------------")
        # print("Support Vectors: " + str(len(self.a)))

        # calculates b intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.support_vectors_y[n]
            self.b -= np.sum(self.a * self.support_vectors_y * K[ind[n], support_vectors])
        if len(self.a) > 0:
            self.b /= len(self.a)
        # calculates the weights vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(number_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.support_vectors_y[n] * self.support_vectors[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, support_vectors_y, support_vectors in zip(self.a, self.support_vectors_y, self.support_vectors):
                    s += a * support_vectors_y * self.kernel(X[i], support_vectors)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        prediction_values, predictions = self.training_result(X)
        return predictions

    def training_result(self, X):
        prediction_values = self.project(X)
        return prediction_values, np.sign(prediction_values)