import numpy as np
from scipy.optimize import minimize
from scipy import stats

def sigmoid(z):
    '''sigmoid函数
    '''
    return 1 / (1 + np.exp(-z))

def sigmoidDerivative(y):
    '''sigmoid求导
    '''
    return np.multiply(y, (1-y))

def initThetas(hiddenNum, unitNum, inputSize, classNum, epsilon):
    '''初始化权值矩阵

    Args:
        hiddenNum 隐含层数目
        unitNum   隐藏单元数目
        inputSize 输入层规模
        classNum  分类数目
        epsilon   epsilon
    
    Returns:
        Thetas 权值矩阵序列
    '''
    # 各隐含层隐藏单元数目 1层，单元数为25
    hiddens = [unitNum for i in range(hiddenNum)]
    # 全部层的隐藏单元数目
    # unit = [[n], [25], [10]]
    units = [inputSize] + hiddens + [classNum]
    Thetas = []
    # 0,n;1,25;2;10
    for idx, unit in enumerate(units):
        if(idx == len(units) - 1): #到输出层结束,输出层无theta矩阵
            break
        nextUnit = units[idx + 1]
        #考虑偏置情况Theta矩阵行列数=S_(j+1),(S_j)+1
        Theta = np.random.rand(nextUnit,unit + 1) * (2 * epsilon) - epsilon
        Thetas.append(Theta) # [[25x(n+1)],[10x26]]
    print(Thetas)
    #Thetas是Theta矩阵的集合形成的列表
    return Thetas

def fp(Thetas, X):
    '''前向传播过程

    Args:
        Thetas 权值矩阵
        X 样本矩阵
    Returns：
        a 各层激励值
    '''
    # theta的行代表下层激励值个数（不加偏置），列代表本层特征数（加偏置）
    # thetas的长度代表theta的个数，即参数矩阵的个数，为2
    layers = range(len(Thetas) + 1)#2 + 1 =3，加上没有Theta的输出层，一共为3层
    layerNum = len(layers)#layers=range(0,3),layerNum=3，3层
    #激活向量序列
    a = list(range(layerNum)) #此处存疑？是否可以直接list(layers)
    # print('a is : ',a)
    # b = list(layers)
    #print('b is :' ,b)
    # 前向传播计算各层输出 
    #a[0] = X.T,a[1]=s(z),z=thetas[0]*a[0],a[0]=X.T
    #求出a的结果仍然是不加偏置的
    for l in layers:
        if l == 0:
            a[l] = X.T # X.T=400x5000
        else:
            # Thetas[0] = 25x401,Thetas[1]=10x26
            z = Thetas[l - 1] * a[l - 1] #z1=25x5000,加偏置26x5000
            a[l] = sigmoid(z)
        # 除输出层外，均需加偏置
        if l != layerNum-1:
            # 除输出层外，加偏置,X.T加一行1
            a[l] = np.concatenate((np.ones((1, a[l].shape[1])),a[l]))
    return a
        
def bp(Thetas, a, y, theLambda):
    '''反向传播过程：

    Args:
        a 激活值
        y 标签
    Retruns:
        D 权值矩阵
    '''
    m = y.shape[0] # 5000样本
    layers = range(len(Thetas) + 1)
    layerNum = len(layers)
    d = list(range(len(layers)))#??直接使用layers不可以吗？
    delta = [np.zeros(Theta.shape) for Theta in Thetas]
    for l in layers[::-1]: #倒序
        if l == 0:
            #输入层不计算误差
            break
        if l == layerNum - 1:
            # 输出层误差
            d[l] = a[l] - y.T# 10x5000 - 1x5000
            # print(d[l].shape)
            # d[2] = 10x5000,d[1]=25x5000,d[0]=400x5000
        else:
            # 忽略偏置
            # print(Thetas[l][:,1:].shape)
            #d[l] = theta.T*d[l+1]*(a[l]倒数)
            d[l] = np.multiply((Thetas[l][:,1:].T * d[l + 1]), sigmoidDerivative(a[l][1:, :]))
    for l in layers[0:layerNum-1]: #两层delta
        delta[l] = d[l + 1] * (a[l].T)# delta[0]=25x26;delta[1]=10x10
    D = [np.zeros(Theta.shape) for Theta in Thetas]
    for l in range(len(Thetas)):
        Theta = Thetas[l]
        # 偏置更新增量
        D[l][:, 0] = (1.0 / m) * (delta[l][0:, 0].reshape(1, -1))
        # 权值更新增量
        D[l][:, 1:] = (1.0 / m) * (delta[l][0:, 1:] + theLambda * Theta[:, 1:])
    return D

def computeCost(Thetas, y, theLambda, X=None, a=None):
    '''计算代价

    Args:
        Thetas 权值矩阵序列
        y 标签集
        theLambda 正则系数
        X 样本集
        a 各层激活值
    Returns:
        J 预测代价
    '''
    m = y.shape[0]
    if a is None:
        a = fp(Thetas, X)
    error = -np.sum(np.multiply(y.T, np.log(a[-1])) + np.multiply((1-y).T, np.log(1-a[-1])))
    # 正规化系数
    reg = -np.sum([np.sum(Theta[:, 1:]) for Theta in Thetas])
    return (1.0 / m) * error + (1.0 / (2 * m)) * theLambda * reg

def unroll(matrixes):
    '''参数展开
    Args:
        matrixes 矩阵
    Return:
        vec 向量
    '''
    vec = []
    for matrix in matrixes:
        vector = matrix.reshape(1, -1)[0]#取得行向量，reshape会多出一个[]，[0]用来去掉[]
        vec = np.concatenate((vec, vector))
    return vec

def roll(vector, shapes):
    '''参数恢复

    Args:
        vector 向量
        shape 矩阵形状列表
    Returns：
        matrixes 恢复的矩阵序列
    '''
    matrixes = []
    begin = 0
    for shape in shapes:
        end = begin + shape[0] * shape[1]
        matrix = vector[begin:end].reshape(shape)
        begin = end
        matrixes.append(matrix)
    return matrixes

def updateThetas(m, Thetas, D, alpha, theLambda):
    '''更新权值

    Args:
        m 样本数
        Thetas 各层权值矩阵
        D 梯度
        alpha 学习率
        theLambda 正规化参数
    
    Returns:
        Thetas 更新后的权值矩阵
    '''
    for l in range(len(Thetas)):
        Thetas[l] = Thetas[l] -alpha * D[l]
    return Thetas

def gradientCheck(Thetas, X, y, theLambda):
    '''梯度校验

    Args:
        Thetas 权值矩阵
        X 样本
        y 标签
        theLambda 正则化系数
    Retruns:
        checked 是否检测通过
    '''
    m, n =X.shape
    # 前向传播计算各个神经元的激活值
    a = fp(Thetas, X)
    # 反向传播计算梯度增量
    D = bp(Thetas, a, y, theLambda)
    # 计算预测代价
    J = computeCost(Thetas,y, theLambda, a=a)
    Dvec = unroll(D)
    #求梯度近似
    epsilon = 1e-4
    gradApprox = np.zeros(Dvec.shape)
    ThetaVec = unroll(Thetas)
    shapes = [Theta.shape for Theta in Thetas]
    for i, item in enumerate(ThetaVec):
        ThetaVec[i] = item - epsilon
        JMinus = computeCost(roll(ThetaVec, shapes), y, theLambda, X=X)
        ThetaVec[i] = item + epsilon
        JPlus = computeCost(roll(ThetaVec, shapes), y, theLambda, X=X)
        gradApprox[i] = (JPlus-JMinus) / (2*epsilon)
    #用欧式距离表示近似程度
    diff = np.linalg.norm(gradApprox-Dvec)
    if diff < 1e-2:
        return True
    else:
        return False
def adjustLabels(y):
    '''校正分类标签

    Args: 
        y 标签集
    Returns:
        yAdjusted 校正后的标签集
    '''
    # 保证标签对类型的标识是逻辑标识
    if y.shape[1] == 1:
        classes = set(np.ravel(y)) # 类的个数，set去重复值
        # print(classes)
        classNum = len(classes) # 例中为10
        minClass = min(classes) # 1
        # print(classNum)
        if classNum > 2:
            # yAdjust 为10x10矩阵，行列为类数
            yAdjusted = np.zeros((y.shape[0], classNum), np.float64)
            for row, label in enumerate(y):
                #把原来y中对应值在矩阵中置为1
                yAdjusted[row, label - minClass] = 1
        else:
            # 二分类
            yAdjusted = np.zeros((y.shape[0], 1), np.float64)
            for row, label in enumerate(y):
                if label != minClass:
                    # 把其中一个置为1
                    yAdjusted[row, 0] = 1.0
        return yAdjusted
    # 如果y的列不为1，则本身即为一个矩阵，无需调整
    return y

def gradientDescent(Thetas, X, y, alpha, theLambda):
    '''梯度下降

    Args:
        X 样本
        y 标签
        alpha 学习率
        theLambda 正规化参数
    Returns:
        J 预测代价
        Thetas 更新后的各层权值矩阵
    '''
    # 样本数，特征数
    m, n = X.shape
    #前向传播计算各神经元的激活值
    a = fp(Thetas, X)
    # 反向传播计算梯度增量
    D = bp(Thetas, a, y, theLambda)
    # 计算预测代价
    J = computeCost(Thetas, y, theLambda, a=a)
    # 更新权值
    Thetas = updateThetas(m, Thetas, D, alpha, theLambda)
    if np.isnan(J):
        J = np.inf
    return J, Thetas


def train(X, y, Thetas=None, hiddenNum=0, unitNum=5, epsilon=1, alpha=1, theLambda=0, precision=0.01, maxIters=50):
    '''网络训练

    Args:
        X 训练样本
        y 标签集
        Thetas 初始化的Thetas,如果为None,由系统随机初始化Thetas
        hiddenNum 隐藏层数目
        epsilon 初始化权值的范围[-epsilon, epsilon]
        alpha 学习率
        theLambda 正规化参数
        precision 误差精度
        maxIters 最大迭代次数
    '''
    # 样本数，特征数
    m, n = X.shape
    #矫正标签集
    y = adjustLabels(y)
    # print(y) y已经经过调整，列值为类数目
    classNum = y.shape[1]
    # 初始化Theta
    if Thetas is None:
        Thetas = initThetas(
            hiddenNum=hiddenNum, # 1
            unitNum = unitNum,   # 25
            inputSize = n,       # n
            classNum = classNum, # 10
            epsilon = epsilon    #  1
        )
    # 先进性梯度校验
    print('Doing Gradient Checking....')
    checked = gradientCheck(Thetas, X, y, theLambda)
    if checked:
        for i in range(maxIters):
            error, Thetas = gradientDescent(
                Thetas, X, y, alpha, theLambda = theLambda)
            if error < precision:
                break
            if error == np.inf:
                break
        if error < precision:
            success = True
        else:
            success = False
        return{
            'error' : error,
            'Thetas' : Thetas,
            'iters' : i,
            'success' : success
        }
    else:
        print('Error: Gradient Checking Failed!!')
        return{
            'error' : None,
            'Thetas' : None,
            'iters' : 0,
            'success' : False
        }

def predict(X, Thetas):
    '''预测函数

    Args: 
        X 样本
        Thetas 训练后得到的参数
    Retruns:
        a 激活向量
    '''
    a = fp(Thetas, X)
    return a[-1]