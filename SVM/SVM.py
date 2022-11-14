import numpy as np

class SVM:

    def __init__(self, lr=0.001, lambda_param = 0.1, max_iter = 1000 ):
        self.lr = lr
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.wt = None
        self.b = None
    
    def fit(self,X,y):
        y_ = np.where(y<=0,-1,1)
        self.wt = np.random.rand(X.shape[1])
        self.b = 0

        for i in range(self.max_iter):
            for idx, x_i in enumerate(X):
                cond = y_[idx] * [np.dat(x_i,self.wt)-self.b] >=1
                if cond:
                    self.wt -= self.lr * (2 * self.lambda_param * self.wt)
                else:
                    self.wt -=  self.lr * (2 * self.lambda_param * self.wt - np.dot(x_i,y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self,X):
        predicition = np.dot(X,self.wt) - self.b
        return predicition



