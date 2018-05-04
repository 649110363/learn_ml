import numpy as np
import time

#cost time cacl
def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc

def loadDataSet(filename):
    '''读取数据集  从文件中获取数据(数据源：机器学习实战)

    Args:
        filename 文件名
    
    Returns:
        X 训练样本
        y 标签矩阵
    '''
    #特征个数，最后一个为标签，故-1
    nuFeat = len(open(filename).readline().split('\t')) - 1
    X = []
    y = []
    file = open(filename)
    for line in file.readlines():
        lineArr = []
        curline = line.strip().split('\t')
        for i in range(nuFeat):
            lineArr.append(float(curline[i]))
        X.append(lineArr)
        y.append(float(curline[-1])) #注意格式转换，y默认格式<U8,会导致图像错误
    return np.mat(X), np.mat(y).T

def h(X, theta):
    return (theta.T*X)[0,0]

def J(X,theta,y):
    m = len(X)
    return (X*theta-y).T * (X*theta-y)/(2*m)

#batch gradient descent
@exeTime
def bgd(X, y, alpha, epsilon, maxLoop):
    '''batch gradient descent 
    
    args:
        X:feature
        y:label
        alpha: learn rate
        epsilon: convergence precision
        maxLoop: max iterations
    '''
    m,n = X.shape
    #init theta
    theta = np.zeros((n,1))
    converged = False
    error = float('inf') # set error = int
    errors = []
    thetas = {} #set theta in a dict
    count = 0
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count < maxLoop:
        if (converged):
            break
        count += 1
        for j in range(n):
            #print(X.dtype, y.dtype)
            #y = y.astype('float64')#X与y格式不同，要转换格式
            deriv = (y - X*theta).T * X[:, j] / m
            theta[j,0] = theta[j,0] + alpha * deriv
            thetas[j].append(theta[j,0])
        error = J(X, theta, y)
        errors.append(error[0,0])
        if error < epsilon:
            converged = True
    return theta, errors, thetas

#Stochastic gradient descent
@exeTime
def sgd(X,y,alpha,epsilon,maxLoop):
    '''随机梯度下降
    Args:
        X 样本数据
        y 样本标签
        alpha 学习率
        epsilon 收敛精度
        maxLoop 最大迭代次数
    '''
    m,n = X.shape
    theta = np.zeros((n, 1))
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    count = 0
    #init thetas
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count < maxLoop:
        if (converged):
            break
        count += 1
        errors.append(float('inf'))
        for i in range(m):#对每个样本来说
            if converged:
                break
            diff = h(X[i].T,theta) - y[i,0]
            for j in range(n):
                theta[j,0] = theta[j,0] - alpha * diff * X[i,j]
                thetas[j].append(theta[j,0])
            error = J(X,theta,y)
            errors[-1] = error[0,0]
            if(error < epsilon):
                converged = True
    return theta, errors, thetas

def JLwr(theta, X, y, x, c):
    '''局部加权线性回归代价函数计算式
    Args:
        theta 参数矩阵
        X 样本矩阵
        y 标签矩阵
        x 待预测输入
        c 权值
    '''
    m, n = X.shape
    summerize = 0
    for i in range(m):
        diff = (X[i] - x) * (X[i] - x).T
        w = np.exp(-diff/(2*c*c))
        predictDiff = np.power(y[i] - X[i]*theta,2)
        summerize = summerize + w*predictDiff
    return summerize

@exeTime
def Lwr(rate, maxLoop, epsilon,X,y,x, c=1):
    '''局部加权线性回归
    Args:
        rate 学习率
        maxLoop 最大迭代次数
        epsilon 预测精度
        X 样本矩阵
        y 标签矩阵
        x 待预测向量
        c 权值
    '''
    m,n = X.shape
    theta = np.zeros((n,1))
    thetas = {}
    error = float('inf')
    errors = []
    count = 0
    converged = False
    for j in range(n):
        thetas[j] = [theta[j,0]]
    #执行批量梯度下降
    while count< maxLoop:
        if converged:
            break
        count = count + 1
        for  j in range(n):
            deriv = (y - X*theta).T*X[:,j]/m
            theta[j,0] = theta[j,0]+rate*deriv
            thetas[j].append(theta[j,0])
        error = JLwr(theta,X,y,x,c)
        errors.append(error[0,0])
        if (error < epsilon):
            converged = True
    return theta,errors,thetas

def standarize(X):
    '''特征标准化

    Args:
        X 样本矩阵
    Retrurns:
        标准化后样本集
    '''
    m,n = X.shape
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0) #计算列的标准差
        if std != 0:
            X[:,j] = (features - meanVal)/std
        else:
            X[:,j] = 0
    return X

def normalize(X):
    '''特征归一化处理

    Args:
        X 样本集
    Returns:
        归一化后的样本集
    '''
    m,n = X.shape
    for j in range(n):
        fetaure = X[:,j]
        minVal = fetaure.min(axis=0)
        maxVal = fetaure.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
            X[:,j] = (fetaure - minVal)/diff
        else:
            X[:,j] = 0
    return X
