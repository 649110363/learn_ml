'''逻辑AND运算
'''
import neural_network as nn
import numpy as np

data = np.mat([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

X = data[:, 0:2]
y = data[:, 2]
print(X, y)

res = nn.train(X, y, hiddenNum=0, alpha=10, maxIters=5000, precision=0.01)
print('Run %d iterations'%res['iters'])
print('Error is: %.4f'%res['error'] )
print('Theta is: ', res['Thetas'][0])