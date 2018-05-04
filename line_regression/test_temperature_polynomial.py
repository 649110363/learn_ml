import regression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

if __name__ == '__main__':
    srcX, y = regression.loadDataSet('data/temperature.txt')

    m, n = srcX.shape
    #二次多项式
    srcX = np.concatenate((srcX[:,0], np.power(srcX[:, 0], 2)), axis=1)

    #特征缩放
    X = regression.standarize(srcX.copy())
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    rate = 0.1
    epsilon = 0.01
    maxLoop = 1000

    result, timeConsumed = regression.bgd(X, y, rate, epsilon, maxLoop)
    theta,errors,thetas = result
    print(X, y, rate, epsilon, maxLoop)
    #打印特征点
    fittingFig = plt.figure()
    title = 'polynomorl with bgd: rate=%.2f, epsilon=%.2f,maxLoop=%d \n time=%ds'%(rate,epsilon,maxLoop,timeConsumed)
    ax = fittingFig.add_subplot(111,title=title)
    trainingSet = ax.scatter(srcX[:,0].flatten().A[0], y[:,0].flatten().A[0])

    #打印拟合曲线
    xx = np.linspace(50, 100, 50)
    xx2 = np.power(xx, 2)
    yHat = []
    for i in range(50):
        normalizedSize = (xx[i]-xx.mean())/xx.std(0)
        normalizedSize2 = (xx2[i]-xx2.mean())/xx2.std(0)
        x = np.matrix([[1,normalizedSize,normalizedSize2]])
        yHat.append(regression.h(x.T,theta))
        #print(x)
        #print(yHat)
    fittingLine, = ax.plot(xx,yHat,color='g')

    ax.set_xlabel('temperature')
    ax.set_ylabel('yield')

    plt.legend([trainingSet, fittingLine], ['Training Set', 'Polynomial Regression'])
    plt.show()

    #打印误差曲线
    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Number of iteration')
    ax.set_ylabel('Cost J')

    plt.show()
