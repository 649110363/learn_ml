#计算给定数据的香农熵

from math import log
import operator

def calcEnt(dataSet):
    '''计算香农熵

    Args:
        dataSet 给定数据集
    
    Returns:
        Ent 香农熵
    '''
    numEntries = len(dataSet)
    labelCount = {}
    for fetVec in dataSet:
        #获取各标签数量
        labelCount[fetVec[-1]] = labelCount.get(fetVec[-1], 0) + 1
    # print(labelCount)
    Ent = 0.0
    #使用k,v遍历必须使用items
    for key,val in labelCount.items():
        prob = float(val)/numEntries
        Ent -= prob * log(prob, 2)
    return Ent

def majorityCnt(classList):
    '''多数表决分类
    
    Args:
        classList 数据集
    Returns：
        sortedClassCount 返回数量最多的类别
    '''
    classCount = {}
    for feaVec in classList:
        classCount[feaVec[-1]] = classCount.get(feaVec[-1], 0) + 1 
    # 一定要加items()
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet, axis, value):
    '''划分数据集
    
    Args:
        dataSet 待划分数据集
        axis 待划分属性
        value 待划分属性对应的值
    
    Returns
        retDataSet 划分后的数据集
    '''
    #不在原数据集操作，而是定义一个新的数据集
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # tempDataSet用来存储减去当前特征值的数据集
            tempDataSet = featVec[:axis]
            tempDataSet.extend(featVec[axis+1:])#所有结果是一个数据集,extend
            retDataSet.append(tempDataSet)# 链接一个一个的数据集，append
    return retDataSet

def chooseBestFeature(dataSet,):
    '''选择最好的数据集划分方式

    Args:
        dataSet 数据集
    
    Returns:
        bestFeature 最佳特征
    '''
    bestFeat = 0.0 #信息增益
    rootEnt = calcEnt(dataSet) #计算根结点的熵
    # 找出所有特征值
    numFea = len(dataSet[0]) - 1 
    for i in range(numFea):
        # FeaList暂存各个特征值，直接在循环中处理，不需保存
        # 不能直接用set去重，因为是单独的int数值，最后才组合成列表
        feaList = [feaVec[i] for feaVec in dataSet]
        feature = set(feaList) #获得特征值
        #按照每个特征值均进行数据集划分，计算信息熵
        feaEnt = 0
        for f in feature:
            # 把该属性下值相同的分为一组
            retDataSet = splitDataSet(dataSet,i,f)
            # 系数
            prob = len(retDataSet)/len(dataSet)
            # 特征结点的熵是每个取值之和
            feaEnt += prob * calcEnt(retDataSet)
        print(feaEnt)
        # 计算熵增益
        gain = rootEnt - feaEnt
        #找出最大的信息增益
        if(bestFeat < gain):
            bestFeat = gain
            bestFeature = i
    return bestFeature
        
def createTree(dataSet, labels):
    '''创建决策树

    Args:
        dataSet 数据集
        labels 标签
    '''    
    # 1.判断是否为叶节点：数据集为同一类或者数据集不包含特征值，后者按多数选择类别
    # 2.非叶节点，进行树的创建
    #     a.选择信息增益最大的特征
    #     b.按特征值划分数据集，存储类别标签，创建决策树（字典存储路径）
    #     c.对每个数据集进行递归操作     
    classList = [example[-1] for example in dataSet]
    # 如果为同一类别，停止划分
    if len(set(classList)) == 1:
    # if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1: #遍历完所有样本，即只剩标签时返回多数类
        return majorityCnt(dataSet)
    bestFeature = chooseBestFeature(dataSet)
    # 求最佳特征的特征值
    featureList = set([feature[bestFeature] for feature in dataSet])
    bestLabel = labels[bestFeature] #取出真正的特征标签
    del(labels[bestFeature]) #删除标签中对应特征值
    myTree = {bestLabel:{}} #存储路径值，最后一次递归返回标签值
    # 对最佳特征的每个特征值递归调用决策树函数
    for value in featureList:
        subLabels = labels[:] #感觉不需要subLabels，因为del操作已经改变了原label列表
        # print(labels)
        myTree[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

def classify(inputTree, featureLabels, testVec):
    '''分类

    Args:
        inputTree 输入树
        featureLabels 标签集
        testVec 待测试集，其值是标签集上特征对应取值
    '''
    firstVec = list(inputTree.keys())[0]
    secondDict = inputTree[firstVec]
    featureIndex = featureLabels.index(firstVec) #testVec代表对应属性的取值
    for key,val in secondDict.items():
        if testVec[featureIndex] == key:
            if(type(val).__name__ == 'dict'):
                classLabel = classify(val, featureLabels, testVec)
            else:
                classLabel = val 
    return classLabel

def createDataSet():
    '''创建数据集用于测试
    '''
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def storeTree(inputTree, filename):
    '''将决策树存储为pickle文件'''
    
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)

def grabTree(filename):
    # 读取文件
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)