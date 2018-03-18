import numpy as np
import datetime
import re
from fastCluster import read
from sklearn.metrics.pairwise import pairwise_distances

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
	dist = 1 - np.mean(NCCc)
	'''
	if shift > 0:
		newy = np.zeros(shift).tolist() + y[1:(len(y)-shift)]
	else:
		newy = y[(1-shift):] + np.zeros(1-shift).tolist()
	'''
	return dist

def readData(name):
	fp = open('C:/study/time-series/'+name+'/dist.txt')
	distArr = []
	trueLabels = []
	i = 0
	maxDist = 0.
	minDist = 1000000000.
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split(',', raw)
		oneRow = [float(itm) for itm in raw if itm != '']
		distArr.append(oneRow[1:])
		tt = oneRow[1:]
		for elem in tt:
			if elem > maxDist: maxDist = elem
			if elem < minDist and elem != 0.: minDist = elem
		trueLabels.append(int(oneRow[0]))
	distMat = np.mat(distArr)
	fp.close()
	return distMat, trueLabels, maxDist, minDist

def storeData(file):
	feats, labels = read(file)
	distMat = pairwise_distances(feats)
	fp = open("C:/study/time-series/"+file+"/dist.txt", 'w')
	for i in range(len(labels)):
		tmp = str(labels[i]) + ','
		for j in range(len(labels)):
			tmp += str(distMat[i, j]) + ','
		tmp += '\n'
		fp.write(tmp)
	fp.close()
	print('finished storing ' + file)
	return

from drawTimeSeries import draw

if __name__ == "__main__":
	print('Storing all data, start time = ' + str(datetime.datetime.now()))
	file = 'cbf'
	print(file)
	draw(file)
	print('finished drawing')
	storeData(file)
	print('end time = ' + str(datetime.datetime.now()))

	print('End of Test!!')
