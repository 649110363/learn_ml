import numpy as np
import time

def exeTime(func):
    '''耗时装饰器
    Args: 
        func 待装饰函数
    Returns:
        newFunc 装饰后的函数
    '''
    def newFunc(*args, **kwargs):
        t0 = time.time()
        back = func(*args, **kwargs)
        return back, time.time() - t0
    return newFunc

def loadDataSet(filename):
    '''读取数据集

    Args:
        filename 文件名
    Returns:
        X 样本集矩阵
        y 标签集矩阵
    '''
    #求x特征个数，划分X y
    numFeat = len(open(filename).readline().split('\t')) - 1
    X = []
    y = []
    for line in open(filename).readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        X.append([1.0,float(lineArr[0]),float(lineArr[1])])
        y.append(float(curLine[-1]))
    return np.mat(X), np.mat(y).T

def sigmoid(z):
    '''sigmoid函数
    '''
    return 1 / (1 + np.exp(-z))

def J(theta, X, y, theLambda=0):
    '''预测代价函数
    Args:
        theta: 参数矩阵
        X: 样本集矩阵
        y: 标签集矩阵
        theLambda: 正则化系数
    Returns:
        误差值
    '''
    m, n = X.shape
    h = sigmoid(X*theta)
    J = (-1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + \
        (theLambda / (2.0 * m)) * np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return(np.inf)
    return J.flatten()[0,0]

@exeTime
def gradient(X,y,options):
    '''随机梯度下降法
    Args:
        X 样本集矩阵
        y 标签集矩阵
        options.rate 学习率
        options.theLambda 正规参数
        options.maxLoop 最大迭代次数
        options.epsilon 收敛精度
        options.method
            - 'sgd' 随机梯度下降法
            - 'bgd' 批量梯度下降法
    Returns:
        (thetas, errors, iterations), timeConsumed
    '''
    m,n = X.shape
    theta = np.ones((n,1))
    thetas = []
    error = float('inf')
    errors = []
    rate = options.get('rate', 0.01)
    epsilon = options.get('epsilon', 0.1)
    maxLoop = options.get('maxLoop', 1000)
    theLambda = options.get('theLambda', 0)
    method = options['method']
    def _sgd(theta):
        converged = False
        for i in range(maxLoop):
            if converged:
                break
            for j in range(m):
                h = sigmoid(X[j]*theta)
                diff = h - y[j]
                theta = theta-rate*(1.0/m)*X[j].T*diff
                error = J(theta,X,y)
                errors.append(error)
                if error < epsilon:
                    converged = True
                    break
                thetas.append(theta)
        return thetas, errors, i+1
    def _bgd(theta):
        for i in range(maxLoop):
            #用dot,因为使用多项式时变成array
            h = sigmoid(X.dot(theta))
            diff = h - y
            #np.r_数组合并
            theta = theta - rate*((1.0/m)*X.T*diff + (theLambda/m)*np.r_[[[0]],theta[1:]])
            error = J(theta,X,y,theLambda)
            errors.append(error)
            if error < epsilon:
                break
            thetas.append(theta)
        return thetas, errors, i+1
    methods ={
        'sgd':_sgd,
        'bgd':_bgd
    }      
    return methods[method](theta)

def OneVsAll(X, y, options):
    '''One-Vs-All 多分类
    Args:
        X: 样本集矩阵
        y: 标签集矩阵
        options: 训练配置
    Returns:
        Thetas 权值矩阵
    '''      
    #类型数np.ravel降维
    classes = set(np.ravel(y))
    #决定决策边界
    Thetas = np.zeros((len(classes),X.shape[1]))
    # 一次选定每种分类对应的样本为正样本，其他样本为负样本，进行逻辑回归
    for idx, c in enumerate(classes):
        newY = np.zeros(y.shape)
        newY[np.where(y == c)] = 1
        result,timeConsumed = gradient(X, newY, options)
        thetas, errors, iterations = result
        Thetas[idx] = thetas[-1].ravel()
    return Thetas

def predictOneVsAll(X,Thetas):
    '''One-Vs=All下的多分类预测
    
    Args:
        X 样本
        Thetas 权值矩阵
    Returns:
        H 预测结果
    '''
    H = sigmoid(Thetas * X.T)
    return H
