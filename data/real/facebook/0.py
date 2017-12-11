import matplotlib.pyplot as plot

def sim(vec1, vec2):
    count = 0
    for i in range(len(vec1)):
        if vec1[i] == 1 and vec2[i] == 1:
            count += 1
    return count

for graphName in ["0"]: #["0", "107", "1684", "1912", "3437", "348", "3980", "414", "686", "698"]:
    edgeList = [map(int, line.split()) for line in open(graphName + ".edges")]
    featureList = [map(int, line.split()) for line in open(graphName + ".feat")]
    
    graph = {}
    for e in edgeList:
        x = e[0]
        y = e[1]
        if x not in graph:
            graph[x] = set()
        if y not in graph:
            graph[y] = set()
        graph[x].add(y)
        graph[y].add(x)
    
    features = {}
    
    for line in featureList:
        features[line[0]] = line[1:]
    
    edgeCount = [0] * 100
    totalCount = [0] * 100
    
    maxSim = 0
    for x in graph.keys():
        for y in graph.keys():
            if x >= y:
                continue
            similarity = sim(features[x], features[y])
            if y in graph[x]:
                edgeCount[similarity] += 1
            totalCount[similarity] += 1
            maxSim = max(maxSim, similarity)
    
    xs = range(maxSim + 1)
    ys = []
    for i in range(maxSim + 1):
        ys.append(edgeCount[i] / float(totalCount[i]) if totalCount[i] != 0 else 0)
    
    print "Edge probabilities:"
    for i in range(len(xs)):
        print "similarity:", xs[i], " probability:", ys[i]
    plot.plot(xs, ys)
    plot.title("Edge probability, graph" + graphName)
    plot.show()
    
    plot.plot(xs, totalCount[0 : maxSim + 1])
    plot.title("Occurrences, graph" + graphName)
    plot.show()
    
    maxDeg = 0
    for x in graph.keys():
        maxDeg = max(maxDeg, len(graph[x]))
    print "Graph", graphName, "has max degree", maxDeg, "and", len(graph), "nodes"
