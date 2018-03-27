import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.cluster import DBSCAN
from sklearn import metrics
from timeSeriesGen import genShift
from fastCluster import read
from rdp import rdp

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def extend(ff):
    newf = []
    lab = [int(itm[0].real) for itm in ff]
    f = [itm[1].real for itm in ff]
   
    curr = -1
    next = -1
    for i in range(len(f)):
        if i == 0: newf.append(f[0]); curr = 0
        else:
            next = i
            step = (f[next] - f[curr]) / (lab[next] - lab[curr])
            for j in range(lab[curr] + 1, lab[next]):
                val = f[curr] + step * (j - lab[curr])
                newf.append(val)
            newf.append(f[next])
            curr = next
    return newf

def angle(a):
	an = []
	for i in range(1, len(a)):
		an.append(np.arctan((a[i][1] - a[i-1][1]) / (a[i][0] - a[i-1][0])) * 180 / np.pi)
	return np.array(an)

def dtwdist(a, b):
    tmp = [[-1. for i in range(len(b)+1)] for j in range(len(a)+1)]
    distmat = np.mat(tmp)
    distmat[0, :] = 100000.
    distmat[:, 0] = 100000.
    distmat[0, 0] = 0.
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = abs(a[i-1] - b[j-1])
            distmat[i, j] = min(distmat[i-1, j], distmat[i, j-1], distmat[i-1, j-1]) + cost
    return distmat[len(a), len(b)] / max(len(a), len(b))

def euclidean(vec1, vec2):
	dist = 0.
	for i in range(len(vec1)):
		dist += pow((vec1[i] - vec2[i]), 2)
	res = np.sqrt(dist)
	return res

def rdpwrap(a, eps):
	b = [[a[i], i] for i in range(len(a))]
	c = rdp(b, eps)
	d = [itm[0] for itm in c]
	return d

def getCorner(a):
	maxdist = 0.
	idx = 0
	n = len(a)
	for i in range(1, n-1):
		#dist(point to the line of two edge nodes)
		#(x2-x1)y+(y1-y2)x+(x1-x2)y1+(y2-y1)x1 / sqr((y1-y2)*2 + (x1-x2)*2)
		rdpdist = abs(((n-1)*a[i] + (a[0]-a[-1])*i - (n-1)*a[0])) / np.sqrt([pow(a[0]-a[-1], 2) + pow(n-1, 2)])
		sumdist = 0.
		for j in range(1, n-1):
			if j < i:
			   sumdist += pow((a[i]-a[0])*j/i + a[0] - a[j],2)
			elif i==j:
			   continue
			else:
				sumdist += pow((a[i]-a[-1])*(n-j)/(n-i) + a[-1] - a[j],2)
			dist = rdpdist * sumdist
		if dist > maxdist: idx = i; maxdist = dist
	return idx, maxdist

def rrdp(c, eps, n):
	a = [[i, c[i]] for i in range(len(c))]
	b = [[100000,a[0]]]
	que = list([a])
	while len(que) != 0:
		c, cd = getCorner(que[0])
		if cd < eps: que.pop(0); continue
		b.append([cd, que[0][c]])
		tmp = que[0]
		que.pop(0)
		if len(tmp[0:c]) > 2:
			que.append(tmp[0:c])
		if len(tmp[c:-1]) > 2:
			que.append(tmp[c:-1])
	b.append([99999, a[-1]])
	b.sort(reverse=True)
	d = [itm[1] for itm in b[:n]]
	d.sort()
	'''
	if len(d) < n:
		print("not enough nodes after rdp")
		exit()
	'''
	a = [itm[1] for itm in b]
	a.sort()
	return [itm[1] for itm in a], a

def PAA(a, n):
	b = []
	m = int(len(a) / n)
	for i in range(m):
		b.append(np.average(a[i*m:(i+1)*m]))
	b.append(np.average(a[n*m:-1]))
	return b

def getDist(a):
    dist = 0.
    dist = abs(((a[-1][0]-a[0][0])*a[1][1] + (a[0][1]-a[-1][1])*a[1][0] + 
                    (a[0][0]-a[-1][0])*a[0][1] + (a[-1][1]-a[0][1])*a[0][0])) /\
    np.sqrt([pow(a[0][1]-a[-1][1], 2) + pow(a[0][0]-a[-1][0], 2)])
    return dist

def distSum(a, idx):
    dist = 0.
    for j in range(1, len(a)-1):
        if j < idx:
            dist += getDist([a[0],a[j],a[idx]])
        elif idx==j:
            continue
        else:
            dist += getDist([a[idx],a[j],a[-1]])
    return dist

def getCorner(a, num):
	extremum = [0]
	for idx in range(1, len(a)-1):
		if a[idx] > a[idx-1] and a[idx] > a[idx+1]: extremum.append(idx)
		if a[idx] < a[idx-1] and a[idx] < a[idx+1]: extremum.append(idx)
	extremum.append(len(a)-1)
	areas = []
	for idx in range(1, len(extremum)-1):
		areas.append([getArea([anew[extremum[idx-1]],extremum[idx-1]],[anew[extremum[idx]],extremum[idx]],[anew[extremum[idx+1]],extremum[idx+1]]), extremum[idx]])
	areas.sort(reverse=True)
	areas[:num]
	return [itm[1] for itm in areas]

def getAllCorner(a):
	extremum = [[0,a[0]]]
	for idx in range(1, len(a)-1):
		if a[idx] > a[idx-1] and a[idx] > a[idx+1]: extremum.append([idx,a[idx]])
		if a[idx] < a[idx-1] and a[idx] < a[idx+1]: extremum.append([idx,a[idx]])
	extremum.append([len(a)-1,a[-1]])
	return extremum


if __name__ == "__main__":
	a = [[1,1],[2,2],[3,6],[4,6],[5,5]]
	getCorner(a)
	print("The end!")
