import numpy as np
from scipy.special import softmax 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __calcgradient(self, X, y, W):
        gradient = np.zeros((3,3))
        y_hat = np.array([softmax(self.W @ X[i]) for i in range(len(y))])
        y_true = np.eye(3)[y]
        l = 0
        for i in range(len(y)):
            for k in np.unique(y):
                l += -(y_true[i][k] * np.log(abs(y_hat[i][k])))
        self.losses.append(l + self.lam * (np.linalg.norm(self.W)**2))
        gradient += (np.array(y_hat - y_true).T @ np.array(X)) + self.lam * 2 * W
        return np.array(gradient)/len(y)
    
    # TODO: Implement this method!
    def fit(self, X, y):
        # Add bias term
        X = np.vstack((np.ones(len(X)),X.T)).T
        # Initialize random weight matrix
        self.W = np.random.rand(X.shape[1], len(np.unique(y)))
        # Create a list to record the losses for use in visualize_loss
        self.losses = []
        self.num_iters = 200000
        for i in range(self.num_iters):
            gradient = self.__calcgradient(X, y, self.W)
            self.W -= self.eta * gradient
        self.losses = np.array(self.losses)
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        # Add bias term
        X_pred = np.vstack((np.ones(len(X_pred)),X_pred.T)).T
        preds = []
        for x in X_pred:
            preds.append(np.argmax(softmax(self.W @ x)))
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.title('Iterations vs. Loss for eta = ' + str(self.eta) + ' & lambda = ' + str(self.lam))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        # set ylim so that our plot is more interpretable
        plt.ylim(0, 20)
        plt.plot(range(self.num_iters),self.losses)
        plt.savefig(output_file + '.png')
        if show_charts:
            plt.show()
        pass