import numpy as np #numpy for linear solving 

def MSE(y_test,y_pred): #Mean_squared_error
    return np.mean(y_test - y_pred)**2

class Linear_Regression:
    def __init__(self,alpha = 0.001,max_iter = 1500): ##learning_rate(alpha) and iterations(max_iter)
        self.alpha = alpha
        self.max_iter = max_iter
        self.wt,self.bias = None,None #weight and bias is set to none
    
    def fit(self,X,y):
        self.wt = np.zeros(X.shape[1]) #no.of weights = no. of features (shape([1])) -> gives number of features. It creates a zero array for each weight
        self.bias = 0 #bias is set to zero, you can also do np.random

        for i in range(self.max_iter):
            y_pred = np.dot(X,self.wt) + self.bias #y_pred = w1*x1 + w2*x2 + ....Wn*xn + bias
            
            dw = 1/X.shape([0]) * np.dot(X.T,(y_pred-y)) # derivative of weight ->dw = 1/m * X.T * (y_pred - y) , dot product will do the summation itself, you can also do 2/m if u want
            db = 1/X.shape([0]) * np.sum(y_pred-y) # derivative of bias ->db = 1/m * sum(y_pred - y)

            self.wt = self.wt - self.alpha*dw #weight_updation -> weight - learning_rate * dw
            self.bias = self.bias - self.alpha*db #bias_updation -> bias - learning_rate * dw

    def predict(self,X):
        y_pred = np.dot(X,self.wt) + self.bias #
        return y_pred