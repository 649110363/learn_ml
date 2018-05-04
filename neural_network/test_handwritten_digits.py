'''手写字符集
'''
import neural_network as nn
import numpy as np
from sklearn import datasets
from scipy.io import loadmat

data = loadmat('data/handwritten_digits.mat')
Thetas = loadmat('data/ex4weights.mat')
print(Thetas['Theta1'].shape, Thetas['Theta2'].shape)
Thetas = [Thetas['Theta1'], Thetas['Theta2']]

# print(data, Thetas)

X = np.mat(data['X'])
y = np.mat(data['y'])
# print(y.shape)
print(X.shape,y.shape)
res = nn.train(X, y, hiddenNum=1, unitNum=25, Thetas=Thetas, precision=0.5)
print('Error is: %.4f'%res['error'])