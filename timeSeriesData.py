import numpy as np
import datetime
import re

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
	fp = open('C:/study/time-series/'+name+'/distMatrix.txt')
	distArr = []
	trueLabels = []
	i = 0
	maxDist = 0.
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split(',', raw)
		oneRow = [float(itm) for itm in raw if itm != '']
		distArr.append(oneRow[1:])
		tt = oneRow[1:]
		for elem in tt:
			if elem > maxDist: maxDist = elem
		trueLabels.append(int(oneRow[0]))
	distMat = np.mat(distArr)
	fp.close()
	return distMat, trueLabels, maxDist

def storeData():
	names = ['ecgfivedays', 'ecg200']
	for name in names:
		feat = []
		labels = []
		if name == 'ecgfivedays':
			feat, labels = fiveDays()
		elif name == 'ecg200':
			feat, labels = ecg200()
		distMat = writeDistMat(name, feat, labels)
		print('finished storing ' + name)
	return

def fiveDays():
	fp = open('C:/study/time-series/ecgfivedays/ecgfivedays.txt')
	features = []
	trueLabels = []
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split(',', raw)
		oneRow = [itm for itm in raw if itm != '']
		trueLabels.append(int(oneRow[0]))
		features.append([float(itm) for itm in oneRow[1:]])
	fp.close()
	return features, trueLabels

def ecg200():
	fp = open('C:/study/time-series/ecg200/ecg200.txt')
	features = []
	label = []
	for line in fp.readlines():
		oneRow = []
		raw = line.replace('\n', '')
		raw = re.split('\t', raw)
		oneRow = [itm for itm in raw if itm != '']
		ll = int(oneRow[-1])
		if ll == 1:
			label.append(2)
		elif ll == -1:
			label.append(1)
		features.append([float(itm) for itm in oneRow[:-1]])
	return features, label

def writeDistMat(name, feat, labels):
	tmp = [[0. for i in range(len(feat))] for i in range(len(feat))]
	distMat = np.mat(tmp)
	for xid, x in enumerate(feat):
		for yid, y in enumerate(feat):
			if xid != yid:
				dist = euclidean(x, y)
				distMat[xid, yid] = dist
	fp = open('C:/study/time-series/'+name+'/distMatrix.txt', 'w')
	for i in range(len(labels)):
		tmp = str(labels[i]) + ','
		for j in range(len(labels)):
			tmp += str(distMat[i, j]) + ','
		tmp += '\n'
		fp.write(tmp)
	fp.close()
	return distMat

if __name__ == "__main__":
	print('Storing all data, start time = ' + str(datetime.datetime.now()))
	storeData()
	print('end time = ' + str(datetime.datetime.now()))

	print('End of Test!!')
