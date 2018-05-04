'''将决策树绘制成图形
'''
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
decisionNode = dict(boxstyle='sawtooth', fc='0.8')# 文本框为锯齿形，边框为0.8
leafNode = dict(boxstyle='round4', fc='0.8')#圆形
arrow_args = dict(arrowstyle='<-') #箭头形状

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''绘制带箭头的注解
    '''
    # 绘图区由createPlot.ax1决定
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    '''找到父节点和子节点的中间位置'''
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]# 起点+（终点-起点）/2
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    '''决策树构造方法
    Args:
        myTree 传入的决策树
        parentPt 根结点坐标
        nodeTxt 文本信息
    '''
    numLeafs = getNumLeafs(myTree) #获取当前根结点下叶子数和深度
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    #算出起点位置，x=1+叶子数/宽度 + 偏移量
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) # 计算父节点和子节点的中间位置
    plotNode(firstStr, cntrPt, parentPt, decisionNode) # 画出具体结点,第一层为根结点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD # 下一层
    for key,val in secondDict.items():
        if type(val).__name__ == 'dict':
            plotTree(val, cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(val, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white') #白色画布
    fig.clf() #清空画布
    axprops = dict(xticks=[], yticks=[]) #存放两个子树
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #无边框，将子树作为参数放进去
    plotTree.totalW = float(getNumLeafs(inTree))#叶子数作为总宽度
    plotTree.totalD = float(getTreeDepth(inTree))#深度作为总深度
    plotTree.xOff = -0.5 / plotTree.totalW #x向左偏移 因为x的初始坐标是0.5，所以用0.5来除
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '') # 调用plotTree函数，文本信息为空
    plt.show()

def getNumLeafs(myTree):
    '''求数的叶节点个数'''
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #第一个属性,python3返回迭代器，加list
    secondDict = myTree[firstStr] #取第一个属性的值作为第二个字典
    for key,val in secondDict.items():
        if type(val).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '''求决策树深度
    '''
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key,val in secondDict.items():
        # print(key)
        if type(val).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: #与获得叶节点的区别
            maxDepth = thisDepth #每层只取最大值，而不是加在一起
    return maxDepth




#createPlot()
