import numpy as np
from scipy import stats

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    
    # Private method to calculate distances between two points
    def __calcdist(self, x1, x2):
        return ((x1[0]-x2[0])/3)**2+(x1[1]-x2[1])**2

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            distances = []
            for X in self.X:
                distances.append(self.__calcdist(x,X))
            k_closest = np.argsort(distances)[0:self.K]
            ys = self.y[k_closest]
            preds.append(stats.mode(ys)[0][0])
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y