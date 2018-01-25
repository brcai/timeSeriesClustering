import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import csv
from sklearn import metrics
import datetime
import random
from sklearn.decomposition import PCA
from timeSeriesGen import genShift

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def crossCorr(x, y):
	correlation = np.correlate(x, y)
	dist = correlation/(np.linalg.norm(x) * np.linalg.norm(y))
	return 1 - dist[0]

def euclidean(vec1, vec2):
	dist = 0.
	for i in range(len(vec1)):
		dist += pow((vec1[i] - vec2[i]), 2)
	res = np.sqrt(dist) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
	#res = np.sqrt(dist)
	return res

def l1Norm(vec1, vec2):
	dist = 0.
	norm = 0.
	for i in range(len(vec1)):
		dist += abs(vec1[i] - vec2[i])
		norm += abs(vec1[i]) + abs(vec2[i]) 
	#res = dist/norm
	res = dist
	return res             

def SBD(x, y):
	lent = pow(2, int(np.log2(2*len(x)-1))+1)
	CC = np.real(np.fft.ifft(np.fft.fft(x, lent) * np.conjugate(np.fft.fft(y,lent))))
	NCCc = CC
	NCCc = CC/(np.linalg.norm(x)*np.linalg.norm(y))
	shift = NCCc.tolist().index(max(NCCc)) - len(x)
	dist = 1 - max(NCCc)
	'''
	if shift > 0:
		newy = np.zeros(shift).tolist() + y[1:(len(y)-shift)]
	else:
		newy = y[(1-shift):] + np.zeros(1-shift).tolist()
	'''
	return dist


def read():
	fp = open('C:/study/time-series/ecgfivedays/blobs.txt')
	features = []
	trueLabels = []
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split(',', raw)
		oneRow = [itm for itm in raw if itm != '']
		features.append([float(itm) for itm in oneRow])
	trueLabels = [-1 for i in range(len(features))]
	return features, trueLabels

'''
def read():
	fp = open('C:/study/time-series/ecgfivedays/ecgfivedays.txt')
	features = []
	label = []
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split(',', raw)
		oneRow = [itm for itm in raw if itm != '']
		label.append(int(oneRow[0]))
		features.append([float(itm) for itm in oneRow[1:]])
	return features, label
'''

class IncDP:
	def getDist(self, x, y):
		if self.dist == 'eu':
			return euclidean(x, y)
		elif self.dist == 'l1':
			return l1Norm(x, y)
		elif self.dist == 'sbd':
			return SBD(x, y)
		return -1.


	def addToMicroCluster(self, x, idx):
		minDist = 1000.
		minMicro = -1
		for key in self.micros:
			dist = self.getDist(x, self.micros[key][1])
			if dist <= self.dc:
				if dist < minDist: minDist = dist; minMicro = key
		if minMicro == -1: 
			if len(self.micros) == 0 : return -1
			elif minDist >= self.dc * 2: return -1
			else: return 1
		n = self.micros[minMicro][0]
		self.micros[minMicro][0] += 1
		tmp = [(self.micros[minMicro][2][i] * n + x[i]) / (n+1) for i in range(len(x))]
		self.micros[minMicro][2] = tmp
		self.micros[minMicro][3].append(idx)		
		return 0

	def addMicroCluster(self, x, idx):
		self.micros[len(self.micros)] = [1, x, x, [idx]]
		return

	def microCluster(self, x, idx):
		added = self.addToMicroCluster(x, idx)
		if added == -1: self.addMicroCluster(x, idx)
		elif added == 1: self.rest.append([idx, x])
		return

	def groupRestData(self):
		for itm in self.rest:
			x = itm[1]
			idx = itm[0]
			minDist = 1000.
			minMicro = -1
			for key in self.micros:
				dist = self.getDist(x, self.micros[key][1])
				if dist < minDist: minDist = dist; minMicro = key
			n = self.micros[minMicro][0]
			self.micros[minMicro][0] += 1
			tmp = [(self.micros[minMicro][2][i] * n + x[i]) / (n+1) for i in range(len(x))]
			self.micros[minMicro][2] = tmp
			self.micros[minMicro][3].append(idx)		
		return

	def macroCluster(self):
		#calculate distMatrix of each centroids
		print('calculating distance matrix!!')
		distMat = [[-1. for i in range(len(self.micros))] for j in range(len(self.micros))]
		distMat = np.mat(distMat)
		densityArr = [[self.micros[key][0], key] for key in self.micros]
		densityArr.sort(reverse = True)
		for i in range(len(self.micros)):
			for j in range(len(self.micros)):
				if i == j: distMat[i, j ] = 0.
				else: distMat[i, j] = SBD(self.micros[i][2], self.micros[j][2])

		#assign delta to each centroids, their row is the number of members
		print('assigning clusters')
		deltas = []
		for i in range(len(densityArr)):
			minDist = 10000.
			delta = -1
			for j in range(0, i):
				if self.micros[densityArr[j][1]][0] >= self.micros[densityArr[i][1]][0]:
					if minDist > distMat[densityArr[i][1], densityArr[j][1]]: minDist = distMat[densityArr[i][1], densityArr[j][1]]; delta = densityArr[j][1]
			if delta != -1:
				self.micros[densityArr[i][1]].append([minDist, delta])
			else:
				self.micros[densityArr[i][1]].append([distMat.max(), -1])

		self.assignCluster(densityArr)
		return

	def plotFeats(self):
		feat = self.feats
		dx = [itm[0] for itm in feat]
		dy = [itm[1] for itm in feat]
		dpcolorMap1 = ['r' for i in range(len(feat))]
		for idx, lab in enumerate(self.trueLabels):
			if lab == 1: dpcolorMap1[idx] = 'b'
		plt.scatter(dx, dy, c=dpcolorMap1, marker='.', s=300, alpha=0.5)
		plt.xlabel(r'$x$', fontsize=25)
		plt.ylabel(r'$y$', fontsize=25)
		plt.show()
		return

	def plotMicro(self):
		dx = [itm[0] for itm in self.feats]
		dy = [itm[1] for itm in self.feats]
		colors = cm.rainbow(np.linspace(0, 1, 30))
		dpcolorMap2 = ['r' for itm in range(len(dx))]
		for key in self.micros:
			for itm in self.micros[key][3]:
				plt.annotate(key, (dx[itm], dy[itm]))
		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=300, alpha=0.5)
		dpcolorMap1 = ['b' for i in range(len(self.micros))]
		ddx = [self.micros[key][1][0] for key in self.micros]
		ddy = [self.micros[key][1][1] for key in self.micros]
		plt.scatter(ddx, ddy, c=dpcolorMap1, marker='.', s=300, alpha=0.5)
		plt.xlabel(r'$x$', fontsize=25)
		plt.ylabel(r'$y$', fontsize=25)
		plt.show()
		return

	def plotRes(self):
		dx = [itm[0] for itm in self.feats]
		dy = [itm[1] for itm in self.feats]
		colors = cm.rainbow(np.linspace(0, 1, 4))
		dpcolorMap2 = [colors[self.labels[itm]] for itm in range(len(self.labels))]
		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=300, alpha=0.5)
		plt.xlabel(r'$x$', fontsize=25)
		plt.ylabel(r'$y$', fontsize=25)
		plt.show()
		return


	def plotDgraph(self, centres):
		dx = [self.micros[itm][0] for itm in self.micros]
		dy = [self.micros[itm][4][0] for itm in self.micros]
		dpcolorMap3 = ['k' for i in range(len(dx))]
		cx = [self.micros[itm][0] for itm in centres]
		cy = [self.micros[itm][4][0] for itm in centres]
		plt.scatter(dx, dy, c=dpcolorMap3, marker='.', s=300, alpha=0.5)
		dpcolorMap1 = ['r' for i in range(len(centres))]
		plt.scatter(cx, cy, c=dpcolorMap1, marker='.', s=1000,edgecolor='k')
		plt.xlabel(r'$\rho$', fontsize=25)
		plt.ylabel(r'$\delta$', fontsize=25)
		plt.show()
		return

	def assignCluster(self, densityArr):
		self.clusters = [-1 for key in self.micros]
		id = 0
		densMulDelta = [[self.micros[key][0]*self.micros[key][4][0], key] for key in self.micros]
		densMulDelta.sort(reverse = True)
		centres = [itm[1] for itm in densMulDelta[:self.k]]
		#self.plotDgraph(centres)
		for elem in densityArr:
			if elem[1] in centres: self.clusters[elem[1]] = id; id += 1
			else: 
				if self.micros[elem[1]][4][1] == -1: 
					print('error, increase dc and try again....\n') 
					return False
				self.clusters[elem[1]] = self.clusters[self.micros[elem[1]][4][1]]
		#assign cluster label for each element
		for key in self.micros:
			for itm in self.micros[key][3]:
				self.labels[itm] = self.clusters[key]
		return True

	def incAssign(self):
		#first micro clustering
		print('start micro clustering!!')
		for idx, feat in enumerate(self.feats):
			self.microCluster(feat, idx)
		self.groupRestData()
		self.writeMicros()
		print('number of micros = ' + str(len(self.micros)))
		#self.plotMicro()
		#macro clustering to output
		print('start macro clustering!!')
		self.macroCluster()

		#self.plotRes()
		return

	def writeMicros(self):
		fp = open('C:\\study\\time-series\\git\\timeSeriesClustering\\micros.txt', 'w')
		for key in self.micros:
			fp.write(str(key)+':\n')
			for itm in self.micros[key][3]:
				fp.write(str(itm)+',')
			fp.write('\n')
		fp.close()
		return

	def run(self):
		self.incAssign()
		amiTmp = metrics.adjusted_mutual_info_score(self.trueLabels, self.labels)
		return amiTmp

	def __init__(self, dc):
		#self.feats, self.trueLabels = read()
		self.feats, self.trueLabels = genShift(2, 0, 0, 1)
		self.k = 2
		self.labels = [-1 for i in range(len(self.feats))]
		self.clusters = {}
		self.dc = dc
		self.micros = {}         #key = id, value = [memmbers number, centre, average, members index, [delta distance, delta centroid]]
		self.rest = []
		self.dist = 'eu'


if __name__ == "__main__":
	
	inst = IncDP(0.006)
	ami = inst.run()
	print(ami)
	
	'''
	print('start time = ' + str(datetime.datetime.now()))
	maxAmi = 0.
	for i in frange(0.01, 0.5, 0.01):
		inst = IncDP(i)
		#inst.plotFeats()
		ami = inst.run()
		if ami > maxAmi: maxAmi = ami
		print('dc = '+str(inst.dc)+' ami = '+str(ami))
	print('max ami = ' + str(maxAmi))
	print('end time = ' + str(datetime.datetime.now()))
	print('End of Test!!')
	
	fp = open('C:\\study\\time-series\\git\\timeSeriesClustering\\tmp.txt', 'w')
	feats,labls = genShift(2,0,0,1)
	for i in feats:
		for idx, j in enumerate(feats):
			fp.write(str(idx)+'-'+str(euclidean(i, j))+',')
		fp.write('\n')
	fp.close()
	'''