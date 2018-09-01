import argparse
import random
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from dataset import trafficDataLoader
from model import GRNN
from utils import Log

timStart = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--taskID', type=int, default=1, help='traffic prediction task id')
parser.add_argument('--alpha', type=int, default=0.1, help='traffic prediction task id')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--dimHidden', type=int, default=32, help='GRNN hidden state size')
parser.add_argument('--truncate', type=int, default=144, help='BPTT length for GRNN')
parser.add_argument('--nIter', type=int, default=2, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--showNum', type=int, default=None, help='prediction plot. None: no plot')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main(opt):
    dataLoader = trafficDataLoader(opt.taskID)

    opt.nNode = dataLoader.nNode
    opt.dimFeature = dataLoader.dimFeature
    data = dataLoader.data        # [b, T, n, d]
    data /= 100
    A = dataLoader.A
    A = opt.alpha * A + np.eye(opt.nNode)

    #--------TEST---------
    #data = data[:, 0]
    #data = data[:, np.newaxis]
    #A = np.array([1.])
    #A = A[:, np.newaxis]
    #opt.nNode = 1
    #------TEST END-------

    data = torch.from_numpy(data)                                           # [b, T, n, d]
    A = torch.from_numpy(A[np.newaxis, :, :])                               # [b, n, n]
    hState = torch.randn(opt.batchSize, opt.dimHidden, opt.nNode).double()  # [b, D, n]
    opt.interval = data.size(1)
    yLastPred = 0

    log = Log(opt, timStart)
    net = GRNN(opt)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    
    net.double()
    print(net)

    if opt.cuda:
        net.cuda()
        criterion.cuda()
        data = data.cuda()
        A = A.cuda()
        hState = hState.cuda()

    if opt.showNum != None:
        plt.figure(1, figsize=(12, 5))
        plt.ion
    
    for t in range(opt.interval - opt.truncate):
        x = data[:, t:(t + opt.truncate), :, :]
        y = data[:, (t + 1):(t + opt.truncate + 1), :, :]

        for i in range(opt.nIter):
            O, hState = net(x, hState, A)
            hState = hState.data
            
            loss = criterion(O, y)
            MSE = criterion(O[:, -1, :, :], y[:, -1, :, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save prediction result
            if i == 0:
                log.prediction[:, t, :, :] = O[:, -1, :, :].data
                log.mseLoss[t] = MSE.data
        
        log.showIterState(t)

        # _, hState = net.propogator(x[:, 0, :, :], hState, A)
        # hState = hState.data

        if opt.showNum != None:
            if t == 0:
                if opt.cuda:
                    plt.plot([v for v in range(opt.truncate)], x[0, :, opt.showNum, 0].cpu().data.numpy().flatten(), 'r-')
                    plt.plot([v + 1 for v in range(opt.truncate)], O[0, :, opt.showNum, 0].cpu().data.numpy().flatten(), 'b-')
                else:
                    plt.plot([v for v in range(opt.truncate)], x[0, :, opt.showNum, 0].data.numpy().flatten(), 'r-')
                    plt.plot([v + 1 for v in range(opt.truncate)], O[0, :, opt.showNum, 0].data.numpy().flatten(), 'b-')
            else:
                if opt.cuda:
                    plt.plot([t + opt.truncate - 2, t + opt.truncate - 1], x[0, -2:, opt.showNum, 0].cpu().data.numpy().flatten(), 'r-')
                    plt.plot([t + opt.truncate - 1, t + opt.truncate], [yLastPred, O[0, -1, opt.showNum, 0]], 'b-')
                else:
                    plt.plot([t + opt.truncate - 2, t + opt.truncate - 1], x[0, -2:, opt.showNum, 0].data.numpy().flatten(), 'r-')
                    plt.plot([t + opt.truncate - 1, t + opt.truncate], [yLastPred, O[0, -1, opt.showNum, 0]], 'b-')
            plt.draw()
            plt.pause(0.5)
            yLastPred = O[0, -1, opt.showNum, 0]

        log.saveResult(t)

    if opt.showNum != None:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main(opt)

