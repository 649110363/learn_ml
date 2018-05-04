import numpy as np
import matplotlib as plt
import time

def exeTime(func):
    '''耗时计算器
    '''
    def newFunc(*args, **kwargs):
        t0 = time.time()
        back = func(*args, **kwargs)
        return back, time.time() - t0 
    return newFunc

def h(theta, X):
    '''预测函数

    Args:
        theta 参数矩阵
        X 样本矩阵
    '''
    return (X * theta)
    # return (theta.T * X)[0, 0]

def J(theta, X, y, theLambda=0):
    '''代价函数

    Args:
        theta 参数矩阵
        X 样本矩阵
        y 标签矩阵
        theLambda 正则化系数

    Returns:
        预测误差
    '''
    m = len(X)
    # np.suqare(theta)求平方；使用np.power(theta, 2)也可以
    # X*theta theta的m代表特征个数,n代表人数
    return (X * theta - y).T * (X * theta - y)/(2*m) + theLambda * np.sum(np.square(theta))/(2*m)

@exeTime
def gradient(X, y, rate=1, maxLoop=50, epsilon=1e-1, theLambda=0, initTheta=None):
    '''批量梯度下降发

    Args:
        X 样本矩阵
        y 标签矩阵
        rate 学习率
        maxLoop 最大迭代次数
        epsilon 收敛精度
        theLambda 正规化系数
        initTheta 初始化矩阵
    
    Returns:
        (theta, errors), timeConsumed
    '''
    m, n = X.shape
    #是否给出初始化矩阵，若给出，则为该矩阵，未给出就初始化
    if initTheta is None:
        theta = np.zeros((n,1))
    else:
        theta = initTheta
    # count = 0
    errors = []
    error = float('inf')
    # converged = False
    for i in range(maxLoop):
        theta = theta + (1.0/m)*rate*((y-X*theta).T * X).T # ??? 
        error = J(theta, X, y, theLambda)
        if np.isnan(error):
            error = np.inf
        else:
            error = error[0,0]
        errors.append(error)
        if (error < epsilon):
            break
    return theta, errors

def standardize(X):
    '''特征标准化处理

    Args:
        X 样本矩阵
    
    Returns:
        标准后的样本集
    '''
    m, n = X.shape
    for i in range(n):
        features = X[:,i]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:,i] = (features - meanVal) / std
        else:
            X[:,i] = 0
    return X

def normalize(X):
    '''特征归一化处理

    Args: 
        X 样本集
    Returns:
        归一化后的样本集
    '''
    m, n = X.shape
    for i in range(n):
        features = X[:,i]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff !=0:
            X[:,i] = (features - minVal) / diff
        else:
            X[:,i] = 0
    return X

def getLearningCurves(X, y, Xval, yval, rate=1, maxLoop=50, epsilon=0.1, theLambda=0):
    '''获得学习曲线

    Args:
        X 样本集
        y 标签集
        Xval 交叉验证集
        yval 交叉验证集标签
    
    Retruns:
        trainErrors 训练误差随样本规模变化
        valErrors 校验验证集误差随样本变化
    '''
    m, n = X.shape
    #初始化训练误差与验证误差
    #训练集训练theta,交叉验证集通过计算代价选择模型
    trainErrors = np.zeros((1, m))
    valErrors = np.zeros((1,m))
    for i in range(m):
        Xtrain = X[0:i+1]
        ytrain = y[0:i+1]
        result,timeConsumed = gradient(
            Xtrain,ytrain,rate=rate, maxLoop=maxLoop, epsilon=epsilon, theLambda=theLambda)
        theta,errors = result
        trainErrors[0, i] = errors[-1]
        valErrors[0, i] = J(theta, Xval, yval, theLambda=theLambda)
    return trainErrors,valErrors

