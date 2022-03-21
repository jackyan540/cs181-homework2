import numpy as np
from scipy.stats import mode
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        self.mle_pi = []
        self.mle_mu = []
        self.mle_Sigma = []
        for k in range(3):
            self.mle_pi.append(np.sum(y==k)/len(y))
            self.mle_mu.append(np.mean(X[y==k],axis=0))
            self.mle_Sigma.append(np.cov(X[y==k].T))
        self.mle_pi = np.array(self.mle_pi)
        self.mle_mu = np.array(self.mle_mu)
        if self.is_shared_covariance:
            self.mle_Sigma = self.mle_pi[0]*self.mle_Sigma[0]+self.mle_pi[1]*self.mle_Sigma[1]+self.mle_pi[2]*self.mle_Sigma[2]
        else:
            self.mle_Sigma = np.array(self.mle_Sigma)
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            y_pred = [0,0,0]
            for k in range(3):
                if self.is_shared_covariance:
                    y_pred[k] += mvn.logpdf(x,mean=self.mle_mu[k],cov=self.mle_Sigma) + np.log(self.mle_pi[k])
                else:
                    y_pred[k] += mvn.logpdf(x,mean=self.mle_mu[k],cov=self.mle_Sigma[k]) + np.log(self.mle_pi[k])
            preds.append(np.argmax(y_pred))
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        neg_ll = 0
        for i in range(len(y)):
            for k in np.unique(y):
                if self.is_shared_covariance:
                    neg_ll += -(int(k==y[i])*(np.log(self.mle_pi[k]) + mvn.logpdf(X[i],mean=self.mle_mu[k],cov=self.mle_Sigma)))
                else:
                    neg_ll += -(int(k==y[i])*
                                (mvn.logpdf(X[i],mean=self.mle_mu[k],cov=self.mle_Sigma[k])+np.log(self.mle_pi[k])))
        return neg_ll