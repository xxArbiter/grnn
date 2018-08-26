import datetime
import torch
import os
import scipy.io as spio

def getTime(begin, end=None):
    if end is None:
        end = datetime.datetime.now()
    timeDelta = end - begin
    return '%d h %d m %d.%ds' % (timeDelta.seconds // 3600, (timeDelta.seconds%3600) // 60, timeDelta.seconds % 60, timeDelta.microseconds)


class Log(object):
    def __init__(self, opt):
        self.prediction = torch.zeros(opt.batchSize, opt.interval - opt.truncate, opt.nNode, opt.dimFeature)
        self.mseLoss = torch.zeros(opt.interval - opt.truncate)
        self.taskID = opt.taskID
        self.interval = opt.interval

    def showIterState(self, t, start):
        if t % 10 == 0:
            print('[Log] %d iteration. MSELoss: %.4f, Train used: %s.' % (t + 1, self.mseLoss[t], getTime(start)))

    def saveResult(self, t):
        if not os.path.exists('result'):
            os.mkdir('result')
        if t % 100 == 0 and t != 0:
            print('[Log] Results saved.')
            spio.savemat('result/result_%d.mat' % self.taskID, {'prediction': self.prediction, 'mseLoss': self.mseLoss, 'iter': t})