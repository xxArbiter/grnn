import numpy as np
import scipy.io as spio


class trafficDataLoader():
    """
    Load traffic data from file. [graph, time series]
    """
    def __init__(self, taskID):
        self.graph = self.graphLoader(taskID)
        self.A = self.formAdjMatrix(self.graph)
        self.data = self.seriesLoader(taskID)
        self.nNode = len(self.graph)

    def graphLoader(self, taskID):
        graph = []
        with open('data/graph_%d.csv' % taskID, 'r') as f:
            for line in f:
                outEdge = line[:-1].split(' ')
                outEdge = outEdge[1:]
                for i in range(len(outEdge)):
                    outEdge[i] = int(outEdge[i])
                graph.append(outEdge)

        return graph

    def seriesLoader(self, taskID):
        data = spio.loadmat('data/data_%d.mat' % taskID)
        data = data['data']
        self.dimFeature = data.shape(2)
        return data

    def formAdjMatrix(self, graph):
        dim = len(graph)
        A = np.zeros([dim, dim])
        for i in range(dim):
            for j in graph[i]:
                A[i, j] = 1
        return A
