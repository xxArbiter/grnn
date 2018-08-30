import datetime
import torch
import os
import scipy.io as spio

def getTime(begin, end=None):
    if end is None:
        end = datetime.datetime.now()
    timeDelta = end - begin
    return '%d h %d m %d.%ds' % (timeDelta.seconds // 3600, (timeDelta.seconds%3600) // 60, timeDelta.seconds % 60, timeDelta.microseconds)

def ms2f(ms):
    ms = float(ms)
    while ms >= 1:
        ms /= 10
    return ms

class Log(object):
    def __init__(self, opt, startTime):
        self.opt = opt
        self.resLength = opt.interval - opt.truncate
        self.startTime = startTime

        self.prediction = torch.zeros(opt.batchSize, self.resLength, opt.nNode, opt.dimFeature)
        self.mseLoss = torch.zeros(self.resLength)

    def showIterState(self, t):
        if (t + 1) % 10 == 0 and self.opt.verbal:
            print('[Log] %d iteration. MSELoss: %.4f, Train used: %s.' % (
                    t + 1,
                    self.mseLoss[t],
                    getTime(self.startTime)))

    def saveResult(self, t):
        if not os.path.exists('result'):
            os.mkdir('result')
        if (t + 1) % 100 == 0 or t == self.resLength - 1 and self.opt.verbal:
            if t == self.resLength - 1:
                print('[Log] Train finished. All results saved! Total used: %s.' % getTime(self.startTime))
            else:
                print('[Log] Results saved.')
            duration = datetime.datetime.now() - self.startTime
            timeStamp = '%d%02d%02d_%02d%02d' % (self.startTime.year, self.startTime.month, self.startTime.day,
                    self.startTime.hour, self.startTime.minute)
            spio.savemat('result/result_%d_%s.mat' % (self.opt.taskID, timeStamp), {
                    'prediction': self.prediction.data.numpy(),
                    'mseLoss': self.mseLoss.data.numpy(),
                    'iter': t + 1,
                    'totalTime': duration.seconds + ms2f(duration.microseconds),
                    'batchSize': self.opt.batchSize,
                    'dimHidden': self.opt.dimHidden,
                    'truncate': self.opt.truncate,
                    'nIter': self.opt.nIter,
                    'lr': self.opt.lr,
                    'cuda': self.opt.cuda,
                    'manualSeed': self.opt.manualSeed})
