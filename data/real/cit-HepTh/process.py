#!/usr/bin/env python

import glob
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix

titles = {}
authors = {}
abstracts = {}

for filename in glob.glob("*/*.abs"):
    section = 0
    inTitle = False
    inAuthors = False
    title = ""
    author = ""
    abstract = ""
    for line in open(filename):
        if line.endswith("\n"):
            line = line.strip()
        if line.startswith("\\\\"):
            section += 1
            continue
        if section == 1:
            if line.startswith("Paper: "):
                number = int(line[14:].lstrip("0"))
            elif line.startswith("Title: "):
                inTitle = True
                line = line[7:]
            elif line.startswith("Author"):
                inTitle = False
                inAuthors = True
                line = line[line.find(":") + 2:]
            elif line.startswith("Comment"):
                inAuthors = False
            if inTitle:
                title += line + " "
            if inAuthors:
                line = line.replace(" and ", ", ")
                author += line + " "
        elif section == 2:
            abstract += line + " "
    titles[number] = title.strip()
    abstracts[number] = abstract.strip()
    authors[number] = author.strip().split(", ")

def output(number):
    print "Number:", number
    print "Title:", titles[number]
    print "Authors:", authors[number]
    print "Abstract:", abstracts[number]

#output(1007)
#output(9412001)
#output(9402127)

def selfCite(x, y):
    for a in authors[x]:
        if a in authors[y]:
            return True
    return False

minYear = 1995
maxYear = 2000

graph = {}
citationCount = defaultdict(int)
for line in open('cit-HepTh.txt'):
    if line.startswith('#') or len(line) == 0:
        continue
    split = line.split("\t")
    x = int(split[0])
    y = int(split[1])
    if not selfCite(x, y):
        citationCount[y] += 1 # x cites y, not self-citation

    xYear = x / 100000
    if xYear > 10:
        xYear += 1900
    else:
        xYear += 2000
    if xYear < minYear or xYear > maxYear:
        continue

    if x not in graph:
        graph[x] = set()
    graph[x].add(y)

# Keep those with more than 50 citations
highlyCited = set(x for x in citationCount.keys() if citationCount[x] > 50)
for x in graph.keys():
    if x not in highlyCited:
        del graph[x]
    else:
        graph[x] &= highlyCited

print "Nodes:", len(graph)
print "Edges:", sum(len(s) for s in graph.values())

edgeList = open('cit-HepTh-subset.edges', 'w')
for x in graph.keys():
    for y in graph[x]:
        edgeList.write(str(x) + " " + str(y) + "\n")

nodeList = list(graph.keys())
contents = []
for x in nodeList:
    content = titles[x] + " " + titles[x] + " " + abstracts[x] # titles weighted 2x
    contents.append(content)

tfidfFile = open('cit-HepTh-subset.tfidf', 'w')
tfidf = TfidfVectorizer()
cx = coo_matrix(tfidf.fit_transform(contents))
for i, j, v in zip(cx.row, cx.col, cx.data):
    tfidfFile.write("(%d, %d), %s\n" % (nodeList[i], j, v))
