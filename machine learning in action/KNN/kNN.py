#要作为包导入是命名首字母必须小写
import numpy as np
import operator
import os

def classify0(inX, dataSet, labels, k):
    '''分类

    Args:
        inX 欲分类样本
        dataSet 数据集
        labels 标签
        k 选取的与样本距离最小的点的个数
    '''
    dataSetSize = dataSet.shape[0]
    #扩展inX行数与dataSetSize一致
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = np.square(diffMat)
    #样本与数据集每个数据差异的平方和
    sqDistances = sqDiffMat.sum(axis=1)
    # 距离是标准差
    distances = np.sqrt(sqDistances)
    #索引排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #前k个样本类别,找出最大值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #get方法找不到置0
        # 也可使用collections中的Counter模块
        # classCount = Counter(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3不能使用iteritems方法，可用items替代,用for循环使用
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount[0][0])
    return sortedClassCount[0][0]

def createDataSet():
    '''创建数据集

    Returns:
        groups 特征值集合
        labels 标签
    '''
    groups = np.array([[1.0, 1.1], [1.0, 1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels

def file2matrix(filename):
    '''将文件转化为矩阵

    Args:
        filename 文件名
    
    Returns:
        returnMat 样本矩阵
        classLabelVector 标签向量
    '''
    #使用with...open后会立即关闭文件，因此要保存缓存
    with open(filename) as f:
        lines = f.readlines()
        numberOfLines = len(lines)
        # print(lines, numberOfLines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = [] # 标签向量
        index = 0
        for line in lines:
            line = line.strip()
            listFormLine = line.split('\t')
            # print(listFormLine)
            returnMat[index,:] = listFormLine[0:3]
            classLabelVector.append(int(listFormLine[-1]))
            # print(classLabelVector)
            index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''归一化处理

    Args:
        dataSet 数据集
    
    Returns:
        normDataSet 归一化后的矩阵
        ranges 矩阵中值的范围
        minVals 最小值向量
    '''
    m, n = dataSet.shape
    minVals = dataSet.min(axis=0) # axis=0代表对列(样本)求最小值
    maxVals = dataSet.max(axis=0) 
    ranges = maxVals - minVals
    normDataSet = dataSet - np.tile(minVals,(m,1)) #不进行扩展也可以
    normDataSet = normDataSet / ranges #不扩展，每行均对ranges求商
    # print(normDataSet, ranges, minVals)
    return normDataSet, ranges, minVals

def datingClassTest():
    '''拆分数据集进行测试'''
    hoRatio = 0.1 #测试集比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m, n = normMat.shape
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier : %d, the real answer is %d' %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            # print(classifierResult, datingLabels[i])
            errorCount += 1
    error = errorCount / numTestVecs
    print('the error is: %f' % error)
    return error

def classifyPerson():
    '''手动输入测试'''
    labelList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('please input percentage of time spent playing games:'))
    ffMiles = float(input('frequent filer miles earned per years:'))
    iceCream = float(input('liters of ine cream consumed per years:'))
    Test = np.array([percentTats, ffMiles, iceCream])
    trainSet, labels = file2matrix('datingTestSet2.txt')
    normalTrainSet, ranges, minVals = autoNorm(trainSet)
    normTest = (Test - minVals) / ranges
    print(normTest)
    label = classify0(normTest,normalTrainSet,labels,3)
    print(labelList[label-1])

def img2vec(filename):
    '''将图像转为向量

    Aegs:
        filename 文件名
    
    Returns:
        testVec
    '''
    with open(filename) as f:
        lines = f.readlines()
    testVec = np.zeros((1,1024))
    loc = 0
    for line in lines:
        line = line.strip() # 去除‘\n’
        for i in line:
            testVec[0, loc] = int(i)
            loc += 1
    return testVec

def handWritingClassTest():
    '''手写输入分类测试
    '''
    labelList = [] #从文件名获取数字
    files = os.listdir('trainingDigits')
    m = len(files)
    trainSet = np.zeros((m, 1024))
    for i in range(m):
        labelList.append(files[i].split('_')[0])
        trainSet[i,:] = img2vec('trainingDigits/' + files[i])
    #训练集处理完成，分析测试集
    testfiles = os.listdir('testDigits')
    testNum = len(testfiles)
    # print(testfiles)
    # 分类
    errorCount = 0.0
    for i in range(testNum):
        testLabel = testfiles[i].split('_')[0]
        testVec = img2vec('testDigits/' + testfiles[i])
        preLabel= classify0(testVec,trainSet,labelList,3)
        print('the predict label: %d, the real label: %d' %(int(preLabel), int(testLabel)))
        if (preLabel != testLabel):
            errorCount += 1
    print('error is : %f' %(errorCount/testNum))


if __name__ == '__main__':
    # datingClassTest()
    # classifyPerson()
    handWritingClassTest()


    


