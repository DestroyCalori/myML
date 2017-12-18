import numpy as np

#如果矩阵可逆，可以使用最小二乘法
class LeastSquare:
    def fit(self,X,y):
        if np.linalg.det(X.T*X) == 0:
            raise ValueError
        self.wights = (X.T*X).I*X.T*y
    def predict(self,test):
        return test * self.wights

#局部加权线性回归
class LWLR:

    def __init__(self,k=1):
        self.k = k

    def fit(self,test,X,y):
        m = X.shape[0]
        wights = np.mat(np.eye((m)))
        n = test.shape[0]
        res = []
        for i in range(n):
            for j in range(m):
                tempMat = test[i] - X[j,:]
                wights[j,j] = np.exp(tempMat*tempMat.T/(-2.0*self.k**2))
            if np.linalg.det(X.T*wights*X) == 0:
                raise ValueError
            ws = (X.T*wights*X).I*X.T*wights*y
            ress = test[i]*ws
            res.append(ress.tolist()[0])
        return res

#岭回归ridgeRegres
class RidgeRegres:
    def __init__(self,lam=0.2):
        self.lam = lam
    def fit(self,X,y):
        tempMat = X.T*X + np.eye((X.shape[1]))*self.lam
        # print()
        # print(tempMat)
        if np.linalg.det(tempMat) == 0:
            raise ValueError
        ws = tempMat.I*X.T*y
        # print(ws)
        self.ws = ws
    def predict(self,test):
        return test*self.ws

#前向逐步线性回归
class stageWise:
    def __init__(self,numIt = 100,eps = 0.1):
        self.numIt = numIt
        self.eps = eps
    def fit(self,X,y):
        m,n = X.shape
        returnMat = np.zeros((self.numIt,n))
        ws = np.zeros((n,1))

        wsTest = ws.copy()
        wsMax = ws.copy()
        for i in range(self.numIt):
            lowestError = np.inf
            for j in range(n):
                for sign in [-1,1]:
                    wsTest = ws.copy()
                    wsTest[j] += self.eps*sign
                    # print(wsTest)
                    yTest = X*wsTest
                    mse = (y-yTest).T*(y-yTest)/len(y)
                    if mse < lowestError:
                        lowestError = mse
                        wsMax = wsTest
            ws = wsMax.copy()
        self.ws = ws

    def predict(self,test):
        return test*self.ws


