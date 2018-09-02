import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn import svm
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
    Y.append(data[:, t+144])            # target[n]

trainX = np.array(X[:-720])             # [train, n, 144]
trainY = np.array(Y[:-720])            # [train, n]
testX = np.array(X[-720:])             # [test, n, 144]

start = time.clock()
prediction = np.zeros((data.shape[0], 720))
y = data[:, -720:]
for n in range(data.shape[0]):
    svr = svm.SVR()
    tmpY = np.reshape(trainY[:, n], trainY.shape[0])
    svr.fit(trainX[:, n, :], np.reshape(trainY[:, n], trainY.shape[0]))
    prediction[n, :] = svr.predict(testX[:, n, :])
    print('Finish %d nodes, MSE: %.4f, used: %.1fs' % (n+1, mean_squared_error(y[n, :], prediction[n, :]), time.clock() - start))

error = y - prediction
MSE = (error**2).mean()
VD = ((error - error.mean())**2).mean()
print('MSE: %.4f, VD: %.4f, used: %.1fs' % (MSE*10000, VD*10000, time.clock() - start))
