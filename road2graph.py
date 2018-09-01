taskID = 1
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
            if int(start) in dic:
                dic[int(start)].append(i)    # Only save the relative index to simplify the indexing
            else:
                dic[int(start)] = [i]

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

print(dic)

with open('data/graph_1.csv', 'w') as of:
    for i, n in enumerate(graph):
        of.write('%d' % i)
        for j in range(len(n)):
            of.write(' %d' % n[j])
        
        of.write('\n')
