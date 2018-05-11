import numpy as np
import re
import feedparser
import operator

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 0代表正常言论，1代表侮辱性言论
    return postingList, classVec

def creatVocabList(dataSet):
    '''创建一个去重后的词向量,一维'''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def bagOfWords2Vec(vocablist, inputSet):
    '''判断词汇表中词是否在文档中出现
    Args:
        vocablist 词汇表
        inputSet 文档,一条记录
    
    Returns:
        returnVec 文档向量，取值为0或1
    '''
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            # returnVec[vocablist.index(word)] = 1
            returnVec[vocablist.index(word)] += 1 # 修改为词袋模型
        else:
            print("The word: %s is not in my vocabulary"% word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '''
    Args:
        trainMatrix 文档矩阵
        trainCategory 标签向量
    Returns:
        p0Vect 正常言论概率向量p(wi|c0)
        p1Vect 侮辱言论概率向量p(wi|c1)
        pAbusive 侮辱言论概率
    '''
    numTrainDocs = len(trainMatrix) # 遍历文档向量集
    numWords = len(trainMatrix[0]) # 文档所有出现的词数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 侮辱言论文档个数占总文档比例
    p0Num = np.ones(numWords) #初始化p0,p1，并进行0概率处理
    p1Num = np.ones(numWords)
    p0Denum = 2.0# 由0->2,0概率处理分母
    p1Denum = 2.0 
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #某个文档向量为侮辱性
            p1Num += trainMatrix[i] #每个词的频率
            p1Denum += sum(trainMatrix[i]) # 所有侮辱词汇数量
        else:
            p0Num += trainMatrix[i]
            p0Denum += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denum) # 防止下溢，取对数
    p0Vect = np.log(p0Num/p0Denum)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    if p0 >= p1:
        return 0
    else:
        return 1

def testingNB():
    # 1.获取数据集
    listOPosts, listClass = loadDataSet()
    # 2. 创建词向量
    myVocabList = creatVocabList(listOPosts)
    # 3. 利用词向量对训练集进行0-1转换，建立训练集
    trainMat = []
    for positionDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, positionDoc))
    # 4.求得p(wi|c0)、p(wi|c1)、p(c)
    p0V,p1V,pAb = trainNB0(np.array(trainMat), np.array(listClass))
    # 5.给定数据集，求类别
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
    print('classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
    print('classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    listofTokens = re.split('\\W*', bigString)
    #或者使用compile
    # reg = re.compile('\\W*');reg.split(bigString)
    return [tok.lower() for tok in listofTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # print(docList,fullText,classList)
    vocablist = creatVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet))) #产生一个随机数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) # 划分数据集，4:1
    trainMat = []; trainClass = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocablist, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocablist, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
        # print(errorCount)
    print('The error rate is: ', float(errorCount) / len(testSet))

def calcMostFeq(vocablist, fullText):
    '''统计词频
    
    Args:
        vocablist 词向量
        fullText 全部词汇组成的文本
    
    Returns:
        sortedFreq[:30] 频率最高的30个词汇
    '''
    freqDict = {}
    for token in vocablist:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key =operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    '''使用RSS数据源作为参数测试'''
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = creatVocabList(docList)
    top30Words = calcMostFeq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0]) #移除高频词，即停用词,本例中数据源信息较少，故作用不明显
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        # print(docList[docIndex])
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0.0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList,docList[docIndex])
        if(classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]):
            errorCount += 1.0
    print('the error rate is: ', errorCount / len(testSet))
    return vocabList, p0V, p1V

def getTopWord(ny, sf):
    '''返回大于某个阈值的所有词'''
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**' * 14)
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    print('NY**' * 14)
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
   # testingNB()
   # spamTest()
   #书中的RSS数据源无法访问，更改了两个数据源
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    # vocablist, pSF, pNY = localWords(ny, sf)
    getTopWord(ny, sf)
