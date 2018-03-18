import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import csv
from sklearn import metrics
import datetime
import random
from timeSeriesData import readData
from rdp import rdp

def dtw(x, y):
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = abs(x[i] - y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    return D1[-1, -1]

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def rdp(x, eps, num):
	tmp = [[i, x[i]] for i in range(len(x))]
	tmpNew = rdp(tmp)
	xNew = [itm[1] for itm in tmpNew]
	return xNew

def euclidean(vec1, vec2):
	dist = 0.
	for i in range(len(vec1)):
		dist += pow((vec1[i] - vec2[i]), 2)
	res = np.sqrt(dist)
	return res

def cosine(vec1, vec2):
	dist = 0.
	for i in range(len(vec1)):
		dist += pow((vec1[i] - vec2[i]), 2)
	res = np.sqrt(dist)
	return res

def SBD(x, y):
	lent = pow(2, int(np.log2(2*len(x)-1))+1)
	CC = np.real(np.fft.ifft(np.fft.fft(x, lent) * np.conjugate(np.fft.fft(y,lent))))
	NCCc = CC
	NCCc = CC/(np.linalg.norm(x)*np.linalg.norm(y))
	shift = NCCc.tolist().index(max(NCCc)) - len(x)
	dist = 1 - max(NCCc)
	return dist

class DP:
	def readMat(self):
		distMat, trueLabels, maxDist, minDist = readData(self.name)
		return distMat, trueLabels, maxDist, minDist

	#calculate p and delta
	def calcParams(self, distMat, dc):
		wordParam = {}
        #calculate p as param[0], neighbours as param[1]
		maxp = 0
		maxData = 0
		maxDist = 0.0
		for i in range(0, len(distMat)):
			cnt = 0
			neighbours = []
			for j in range(0, len(distMat)):
				if i!=j:
					tmp = distMat[i, j]
					tmpDist = 0.
					if tmp < dc: 
						cnt += tmp
					if tmp > maxDist: maxDist = tmp
			wordParam[i] = [cnt, neighbours]
			if maxp < cnt: maxp = cnt; maxData = i
		#calculate delta as param[2], nearest higher density point j
		for i in range(0, len(distMat)):
			minDelta = maxDist
			affiliate = -1
			for j in range(0, len(distMat)):
				if wordParam[j][0] > wordParam[i][0]: 
					#SDB distance
					tmp = distMat[i, j]
					if minDelta > tmp: 
						minDelta = tmp
						affiliate = j
			wordParam[i].extend([minDelta, affiliate])
		
		return wordParam

	def plotDgraph(self, wordParams, centres):
		dx = [wordParams[itm][0] for itm in wordParams]
		dy = [wordParams[itm][2][0] for itm in wordParams]
		dpcolorMap3 = ['k' for i in range(len(dx))]
		#cx = [self.micros[itm][0] for itm in centres]
		#cy = [self.micros[itm][4][0] for itm in centres]
		plt.scatter(dx, dy, c=dpcolorMap3, marker='.', s=300, alpha=0.5)
		#dpcolorMap1 = ['r' for i in range(len(centres))]
		#plt.scatter(cx, cy, c=dpcolorMap1, marker='.', s=1000,edgecolor='k')
		plt.xlabel(r'$\rho$', fontsize=25)
		plt.ylabel(r'$\delta$', fontsize=25)
		plt.show()
		return


	def assignCluster(self, wordParams, centers, dc):
		self.plotDgraph(wordParams, centers)
		pRank = [[wordParams[word][0], word] for word in wordParams]
		pRank.sort(reverse = True)
		wordCluster = {word:-1 for word in wordParams}
		id = 0
		centre2cluster = {}
		for p in pRank:
			if wordCluster[p[1]] == -1: 
				if p[1] in centers: wordCluster[p[1]] = id; centre2cluster[p[1]] = id; id += 1
				else: 
					if wordParams[p[1]][3] == -1: 
						#print('error, increase dc and try again....\n') 
						return wordCluster, False
					wordCluster[p[1]] = wordCluster[wordParams[p[1]][3]]
		return wordCluster, True

	def getCenters(self, wordParams, num):
		centers = []
		pRank = []
		for itm in wordParams:
			pRank.append([wordParams[itm][2] * wordParams[itm][0], itm])
		pRank.sort(reverse = True)
		if num > len(pRank): return centers
		centers = [itm[1] for itm in pRank[0:num]]
		return centers

	def run(self, dc, k):
		#distMat, trueLabels, maxDist, minDist = self.readMat()
		#distMat, trueLabels, maxDist, minDist = 
		realDc = (maxDist - minDist)*dc + minDist
		#print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(distMat, realDc)
		centers = self.getCenters(wordParams, k)
		if len(centers) == 0: return wordCluster, label, arsTmp, amiTmp
		wordCluster, valid = self.assignCluster(wordParams, centers, realDc)
		labels = [wordCluster[i] for i in range(len(distMat))]
		amiTmp = metrics.adjusted_mutual_info_score(trueLabels, labels)

		return amiTmp

	def __init__(self, name):
		self.name = name
		self.feats, self.labels = read(name)


from fastCluster import read

def normlize(feat):
	new = []
	avg = 0.
	var = 0.
	for itm in feat:
		avg += itm
	avg = avg / len(feat)
	for itm in feat:
		var += pow(itm - avg, 2)
	var = np.sqrt(var) / len(feat)
	new = [(itm - avg) / var for itm in feat]
	return new

if __name__ == "__main__":
	base1 = [1.2807,1.5708,1.9153,2.3362,2.7351,2.9284,2.8112,2.4533,2.0379,1.6965,1.4205,1.1254,0.75797,0.35345,-0.002452,-0.23449,-0.31305,-0.27555,
	   -0.20967,-0.21105,-0.31069,-0.43958,-0.49433,-0.4358,-0.33923,-0.31363,-0.38318,-0.47433,-0.4964,-0.4357,-0.35011,-0.28571,-0.25934,-0.27502,
	   -0.33756,-0.41314,-0.4167,-0.31361,-0.20237,-0.2565,-0.50839,-0.76383,-0.78701,-0.57426,-0.39353,-0.48503,-0.80027,-1.0652,-1.0917,-0.96191,
	   -0.87657,-0.9195,-1.0095,-1.0647,-1.1147,-1.2139,-1.3277,-1.3543,-1.2686,-1.1697,-1.1631,-1.2366,-1.2731,-1.1908,-1.0229,-0.86577,-0.78111,
	   -0.7522,-0.72919,-0.66635,-0.5354,-0.33547,-0.10215,0.087808,0.17387,0.16455,0.13113,0.14911,0.23193,0.35334,0.49033,0.62967,0.7392,0.75483,
	   0.66171,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
	base2 = [-0.19441,-1.1007,-1.3216,-0.81488,-0.34072,-0.38421,-0.6235,-0.64221,-0.55257,-0.58375,-0.67414,-0.70857,-0.7342,-0.76711,-0.74069,
	   -0.69701,-0.73889,-0.84595,-0.90055,-0.88965,-0.88328,-0.92933,-1.0131,-1.1151,-1.2142,-1.2802,-1.2905,-1.2989,-1.4081,-1.5917,-1.6842,
	   -1.6191,-1.5579,-1.6148,-1.6867,-1.6488,-1.5526,-1.4599,-1.3188,-1.1241,-0.98348,-0.91711,-0.76622,-0.4358,-0.062335,0.14746,0.20885,
	   0.29907,0.4815,0.61784,0.65201,0.70057,0.82487,0.92154,0.97288,1.0792,1.1797,1.1044,0.92291,0.86882,0.92704,0.92672,0.88343,0.89989,0.89992,
	   0.84429,0.87548,0.98458,0.91121,0.64469,0.54624,0.74881,0.9048,0.76475,0.56709,0.63115,0.91369,1.16,1.25,1.2581,1.2938,1.3502,1.3011,1.0883,
	   0.86683,0.81278,0.85455,0.81905,0.76498,0.8761,1.0859,1.1447,1.022,0.92983,0.96388,1.0142]
	aa = normlize(base2).copy()
	tmp1 = normlize(base1).copy()
	tmp2 = tmp1[10:] + tmp1[:10]
	'''
	print('DP: start time = ' + str(datetime.datetime.now()))
	inst = DP('cbf')
	dtwDist(inst.feats[0], inst.feats[1])
	maxAmi = 0.
	dc = 0.
	print('start clustering')
	for i in frange(0.1, 1, 0.05):
		amiTmp = inst.run(i, 3)
		print(amiTmp)
		if maxAmi < amiTmp: maxAmi = amiTmp; dc = i
	print("best ami = "+str(maxAmi)+' best dc = '+str(dc))
	print('end time = ' + str(datetime.datetime.now()))
	'''
	print('End of Test!!')
	
