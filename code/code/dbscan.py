import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.cluster import DBSCAN
from sklearn import metrics
from timeSeriesGen import genShift
from fastCluster import read
from rdp import rdp
from kmeans import extend
from kmeans import dtwdist
from kmeans import angle

def getCorner(a):
	maxdist = 0.
	idx = 0
	for i in range(1, len(a)-1):
		#(x2-x1)y+(y1-y2)x+(x1-x2)y1+(y2-y1)x1 / sqr((y1-y2)*2 + (x1-x2)*2)
		dist = ((a[-1][0]-a[0][0])*a[i][1] + (a[0][1]-a[-1][1])*a[i][0] + (a[0][0]-a[-1][0])*a[i][1] + (a[-1][1]-a[0][1])*a[i][0]) /\
			np.sqrt([pow(a[0][1]-a[-1][1], 2) + pow(a[0][0]-a[-1][0], 2)])
		if dist > maxdist: idx = i
	return idx

def rrdp(a, n):
	b = [a[0]]
	#while len(b) < n-1:


	return

def SBD(x, y):
	lent = pow(2, int(np.log2(2*len(x)-1))+1)
	CC = np.real(np.fft.ifft(np.fft.fft(x, lent) * np.conjugate(np.fft.fft(y,lent))))
	NCCc = CC
	NCCc = CC/(np.linalg.norm(x)*np.linalg.norm(y))
	shift = NCCc.tolist().index(max(NCCc)) - len(x)
	dist = 1 - max(NCCc)
	return dist

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def euclidean(vec1, vec2):
	dist = 0.
	for i in range(len(vec1)):
		dist += pow((vec1[i] - vec2[i]), 2)
	res = np.sqrt(dist)
	return res


class dbscan:
	def calcP(self, distMat, eps):
		pRank = []
		data2P = [0. for i in range(len(distMat))]
		for idx, itm in enumerate(distMat):
			cnt = 0.
			for j in itm:
				if j <= eps: cnt += 1;
			pRank.append([cnt, idx])
			data2P[idx] = cnt
		pRank.sort(reverse=True)
		return pRank, data2P

	def assignCluster(self, dataVecs, epsRaw, minSp):
		visited = {i: False for i in range(len(dataVecs))}
		currentC = -1
		currentD = 0.
		distMat, maxDist = self.calcDists(dataVecs)
		eps = epsRaw
		pRank, data2P = self.calcP(distMat, eps)
		clusterId = 0
		clusters = []
		data2Cluster = [-1 for i in range(len(dataVecs))]

		for data in pRank:
			idx = data[1] 
			if visited[idx] == True: continue
			neighbour = self.getEpsNeighbour(idx, distMat, eps)
			if len(neighbour) < minSp: continue
			visited[idx] = True
			coreList = [idx]
			clusters.append([idx])
			data2Cluster[idx] = clusterId
			while len(neighbour) != 0:
				newNeighbour = []
				for core in neighbour:
					if visited[core] == True: continue
					clusters[clusterId].append(core)
					data2Cluster[core] = clusterId
					visited[core] = True
					coreNeighbour = self.getEpsNeighbour(core, distMat, eps)
					coreDense = len(coreNeighbour)
					common = self.commonNeighbours(core, idx, distMat, eps)
					if len(common) >= minSp:
						adjustedNeighbour = []
						adjustedNeighbour = self.getEpsNeighbour(core, distMat, eps)
						for itm in adjustedNeighbour:
							if itm not in newNeighbour and visited[itm] != True: newNeighbour.append(itm)
				neighbour = newNeighbour
				test = [idx for idx in visited if visited[idx] == True]
			clusterId += 1

		return data2Cluster

	def commonNeighbours(self, a, b, distMat, eps):
		an = self.getEpsNeighbour(a, distMat, eps)
		bn = self.getEpsNeighbour(b, distMat, eps)
		common = []
		for i in an:
			if i in bn: common.append(i)
		return common

	def getEpsNeighbour(self, i, distMat, eps):
		neighbour = []
		for idx, itm in enumerate(distMat[i]):
			if idx != i and itm <= eps: neighbour.append(idx)
		return neighbour

	def calcDists(self, dataVecs):
		distMat = []
		tmpSet = set()
		for i in range(len(dataVecs)):
			tmp = []
			for j in range(len(dataVecs)):
				dist = dtwdist(dataVecs[i], dataVecs[j])
				tmp.append(dist)
				tmpSet.add(dist)
			distMat.append(tmp)
		rankedDist = list(tmpSet)
		rankedDist.sort()
		return distMat, rankedDist[-1]

	#density peak clustering flow
	def run(self, feats, eps, minSp):
		dataVecs = feats
		data2Cluster = self.assignCluster(dataVecs, eps, minSp)
		#self.plotCluster(data2Cluster, dx, dy, centers)
		label = [data2Cluster[i] for i in data2Cluster]
		return data2Cluster, label

if __name__ == "__main__":
	print('DBSCAN: ')
	feats, labs = read('cbf')
	newfeats = []
	for idx, feat in enumerate(feats):
		tt = np.fft.fft(feat)
		#plt.plot(tt)
		ttt = []
		for i in range(len(tt)):
			if abs(i) > 10:
				ttt.append(0)
			else:
				ttt.append(tt[i])
		bb = np.fft.ifft(ttt)
		aa = [[i, bb[i].real] for i in range(len(bb))]
		bb = rdp(aa, epsilon=0.6)
		cc = angle(bb)

		newfeats.append(cc)
	#newfeats = np.array(newfeats)
	max = 0.
	for i in frange(0.1, 8, 0.5):
		dbs = dbscan()
		tt, dlabel = dbs.run(newfeats, i, 4)
		ars = metrics.adjusted_mutual_info_score(dlabel, labs)
		if ars > max: max = ars
		print(str(i)+": "+str(ars))
	print('max nmi : ' + str(max))

	'''
	numbs = [0, 0, 0]
	for i in range(len(labs)):
		if labs[i] == 1 and numbs[0] <= 100: numbs[0] += 1; modfeats.append(feats[i]); newlabs.append(labs[i])
		elif labs[i] == 2 and numbs[1] <= 100: numbs[1] += 1; modfeats.append(feats[i]); newlabs.append(labs[i])
		elif labs[i] == 3 and numbs[2] <= 100: numbs[2] += 1; modfeats.append(feats[i]); newlabs.append(labs[i])

	sig = 0.5
	newfeats = []
	for idx, feat in enumerate(modfeats):
		tmp = [[feat[i], i] for i in range(len(feat))]
		newfeats.append(rdp(tmp, epsilon=sig))
	'''
	'''
	distMat = []
	for i in range(len(newfeats)):
		tmp = []
		for j in range(len(newfeats)):
			tmp.append(euclidean(feats[i], feats[j]))
		distMat.append(tmp)
	print("finish calculating distance matrix")
	'''
	'''
	max = 0.
	for i in frange(1, 20, 0.5):
		k = 10
		print(k)
		inst = dbscan()
		a, b = inst.run(feats, i, k)
		nmi = metrics.adjusted_mutual_info_score(labs, b)
		if nmi > max: max = nmi
		print(str(i)+": "+str(nmi))
	print('max nmi : ' + str(max))
	'''
	'''
	newfeats = []
	for feat in feats:
		tt = np.fft.fft(feat)
		#plt.plot(tt)
		ttt = []
		for i in range(len(tt)):
			if abs(tt[i]) < 12:
				ttt.append(0)
			else:
				ttt.append(tt[i])
		bb = np.fft.ifft(ttt)
		aa = [[i, bb[i]] for i in range(len(bb))]
		bb = rdp(aa, epsilon=0)
		cc = extend(bb)
		newfeats.append(cc)

	max = 0.
	for i in frange(0.1, 20, 0.5):
		dbs = DBSCAN(eps=i, min_samples=5).fit(feats)
		dlabel = dbs.labels_
		nmi = metrics.adjusted_mutual_info_score(labs, dlabel)
		if nmi > max: max = nmi
		print(str(i)+": "+str(nmi))
	print('max nmi : ' + str(max))
	'''
	print('The end...')
