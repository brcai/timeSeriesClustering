import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import csv
from sklearn import metrics
import datetime
import random
from timeSeriesData import readData

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

class SDBDP:
	def calcDistMatrix(self, feat, labels):
		tmp = [[0. for i in range(len(feat))] for i in range(len(feat))]
		distMat = np.mat(tmp)
		for xid, x in enumerate(feat):
			for yid, y in enumerate(feat):
				if xid != yid:
					dist = SBD(x, y)
					distMat[xid, yid] = dist
		fp = open('C:/study/time-series/ecgfivedays/distMatrix.txt', 'w')
		for i in range(len(labels)):
			tmp = str(labels[i]) + ','
			for j in range(len(labels)):
				tmp += str(distMat[i, j]) + ','
			tmp += '\n'
			fp.write(tmp)
		fp.close()
		return

	def readMat(self, name):
		distMat, trueLabels, maxDist = readData(name)
		return distMat, trueLabels, maxDist

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

	def assignCluster(self, wordParams, centers, dc):
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

	def run(self, dc, k, name):
		distMat, trueLabels, maxDist = self.readMat(name)
		realDc = maxDist*dc
		#print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(distMat, realDc)
		centers = self.getCenters(wordParams, k)
		if len(centers) == 0: return wordCluster, label, arsTmp, amiTmp
		wordCluster, valid = self.assignCluster(wordParams, centers, realDc)
		labels = [wordCluster[i] for i in range(len(distMat))]
		amiTmp = metrics.adjusted_mutual_info_score(trueLabels, labels)

		return amiTmp

if __name__ == "__main__":
	print('start time = ' + str(datetime.datetime.now()))
	inst = SDBDP()
	maxAmi = 0.
	dc = 0.
	for i in frange(0.05, 1, 0.05):
		amiTmp = inst.run(i, 2, 'ecgfivedays')
		print(amiTmp)
		if maxAmi < amiTmp: maxAmi = amiTmp; dc = i
	print("best ami = "+str(maxAmi)+' best dc = '+str(dc))
	print('end time = ' + str(datetime.datetime.now()))

	print('End of Test!!')


