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
        selSegs = []
        edges = []
        dic = {}

        with open('data/selSegs_%d.csv' % taskID, 'r') as f:
            for line in f:
                selSegs.append(int(line[:-1]))

        with open('data/segment.csv', 'r') as f:
            i = 0
            for j, line in enumerate(f):
                if j == selSegs[i]:
                    _, start, end = line[:-1].split(',')
                    edges.append([int(start), int(end)])
                    # Form the out-degree singly chain list
                    if start in dic:
                        dic[start].append(i)    # Only save the relative index to simplify the indexing
                    else:
                        dic[start] = [i]

                    if i == len(selSegs) - 1:
                        break
                    else:
                        i += 1
                        
        # Form the out-degree singly chain list
        graph = []
        for i in range(len(selSegs)):
            if edges[i][1] in dic:
                graph.append(dic[edges[i][1]])
            else:
                graph.append([])

        return graph

    def seriesLoader(self, taskID):
        data = spio.loadmat('data/selTraffic_%d.mat' % taskID)
        data = data['selTraffic']
        return data

    def formAdjMatrix(self, graph):
        dim = len(graph)
        A = np.zeros([dim, dim])
        for i in range(dim):
            for j in graph[i]:
                A[i, j] = 1
        return A
