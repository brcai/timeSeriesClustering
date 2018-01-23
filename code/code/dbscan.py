import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.cluster import DBSCAN
from sklearn import metrics
from timeSeriesGen import genShift

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


class dbscan:
	def calcP(self, distMat, eps):
		pRank = []
		data2P = [0. for i in range(len(distMat))]
		for idx, itm in enumerate(distMat):
			cnt = 0.
			for j in itm:
				if j <= eps: cnt += 1;
				#if j <= eps: cnt +=np.exp(-j**2/eps**2)
			pRank.append([cnt, idx])
			data2P[idx] = cnt
		pRank.sort(reverse=True)
		return pRank, data2P

	def assignCluster(self, dataVecs, epsRaw, minSp, lasso, ifNormal):
		visited = {i: False for i in range(len(dataVecs))}
		centers = []
		currentC = -1
		currentD = 0.
		distMat, maxDist = self.calcDists(dataVecs)
		eps = epsRaw*maxDist
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
			center = idx
			centerD = 0
			baseD = data[0]
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
					if coreDense >= minSp:
						if coreDense > centerD: centerD = coreDense; center = core
						adjustedNeighbour = []
						if ifNormal: adjustedNeighbour = self.getEpsNeighbour(core, distMat, eps)
						else: adjustedNeighbour = self.getEpsNeighbour(core, distMat, eps*(data2P[core]/baseD)*lasso)
						for itm in adjustedNeighbour:
							if itm not in newNeighbour and visited[itm] != True: newNeighbour.append(itm)
				neighbour = newNeighbour
				test = [idx for idx in visited if visited[idx] == True]
				#self.plotCluster(data2Cluster, [data[0] for data in dataVecs], [data[1] for data in dataVecs], test)
			clusterId += 1
			centers.append(center)

		return data2Cluster, centers

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
				dist = stpw.euclidean(dataVecs[i], dataVecs[j])
				tmp.append(dist)
				tmpSet.add(dist)
			distMat.append(tmp)
		rankedDist = list(tmpSet)
		rankedDist.sort()
		return distMat, rankedDist[-1]

	#density peak clustering flow
	def run(self, dir, eps, minSp, lasso, kl):
		dx, dy, id, num = self.read(dir)
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		data2Cluster, centers = self.assignCluster(dataVecs, eps, minSp, lasso, False)
		#self.plotCluster(data2Cluster, dx, dy, centers)
		label = [data2Cluster[i] for i in data2Cluster]
		return data2Cluster, label

if __name__ == "__main__":
	feats, labs = genShift(2, 0, 0, 1)
	for i in frange(0.01, 2.6, 0.05):
		dbs = DBSCAN(eps=i, min_samples=1).fit(feats)
		dlabel = dbs.labels_
		print(metrics.adjusted_mutual_info_score(labs, dlabel))
	print('The end...')
