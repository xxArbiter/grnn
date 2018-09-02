import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from dataset import trafficDataLoader
import matplotlib.pyplot as plt
import scipy.io as spio

taskID = 1

dataLoader = trafficDataLoader(taskID)
data = dataLoader.data[0, :, :, 0].transpose()  # [n, T]
data /= 100

X = []
Y = []
for t in range(data.shape[1] - 144):    # T - truncate
    X.append(data[:, t:t+144])          # annotation[n, 144]
    Y.append(data[:, t+144])            # target[n, 1]

trainX = np.array(X[:-720])             # [train, n, 144]
trainY = np.array(Y[:-720])            # [train, n, 1]
testX = np.array(X[-720:])             # [test, n, 144]

start = time.clock()
prediction = np.zeros((data.shape[0], 720))
y = data[:, -720:]
for n in range(data.shape[0]):
    gbrt = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=0, loss='ls')
    gbrt.fit(trainX[:, n, :], np.reshape(trainY[:, n, 0], trainY.shape[0]))
    prediction[n, :] = gbrt.predict(testX)
    print('Finish %d nodes, MSE: used: %.1fs' % (n+1, mean_squared_error(y[n, :], prediction[n, :]), time.clock() - start))

error = y - prediction
MSE = (error**2).mean()
VD = ((error - error.mean())**2).mean()
print('MSE: %.4f, VD: %.4f, used: %.1fs' % (MSE, VD, time.clock() - start))
