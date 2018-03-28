from funcs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from fastCluster import read
import pandas as pd


class newdist:
	def __init__(self, num, p):
		self.num = num
		self.p = p

	def getDist(self, a, b):
		dist = 0.
		atmp = self.fftTrans(a)
		btmp = self.fftTrans(b)

		acorner = getAllCorner(atmp)
		#plt.plot([itm[0] for itm in acorner], [itm[1] for itm in acorner])
		#plt.show()
		bcorner = getAllCorner(btmp)
		#plt.plot([itm[0] for itm in acorner], [itm[1] for itm in acorner])
		#plt.show()

		#dist = self.extDist(acorner, bcorner)
		dist = dtwdist([itm[1] for itm in acorner], [itm[1] for itm in bcorner])
		return dist

	def fftTrans(self, a):
		atmp = np.fft.fft(a)
		afft = [itm.real for itm in np.fft.ifft(np.append(atmp[:self.num], [0 for i in range(len(a)-self.num)]))]
		return afft

	'''
	def pdtw(a, b, p):
		dist = 0.
		anew = [(1-p)*itm[0] + p*itm[1] for itm in a]
		bnew = [(1-p)*itm[0] + p*itm[1] for itm in b]
		dist = dtwdist(anew, bnew)
		return dist
	'''
	
	def fitline(self, start, end, inter):
		val = 0.
		val = (end[1]-start[1]) / (end[0]-start[0]) * (inter[0]-start[0])
		return val

	def showFeat(self, ain):
		atmp = self.fftTrans(ain)
		a = getAllCorner(atmp)
		return a


	def extDist(self, a,b):
		dist = 0.
		anew = [a[0][1]]
		bnew = [b[0][1]]
		i = 1
		j = 1
		n = len(a) - 1
		m = len(b) - 1
		while i != n or j != m:
			if a[i][0] < b[j][0]:
				bnew.append(self.fitline(b[j-1],b[j],a[i]))
				anew.append(a[i][1])
				i += 1
				continue
			if a[i][0] > b[j][0]:
				anew.append(self.fitline(a[i-1],a[i],b[j]))
				bnew.append(b[j][1])
				j += 1
				continue
			else:
				bnew.append(b[j][1])
				anew.append(a[i][1])
				i += 1
				j += 1
				continue
		anew.append(a[-1][1])
		bnew.append(b[-1][1])
		dist = euclidean(anew, bnew) / (len(anew))
		return dist

if __name__ == "__main__":
	feats, labs = read('cbf')
	distCal = newdist(5,0.1)
	'''
	same = distCal.getDist(feats[0],feats[1])
	diff = distCal.getDist(feats[1],feats[3])
	sameeu = euclidean(feats[0],feats[1])
	diffeu = euclidean(feats[1],feats[3])
	
	'''
	a = []
	distmat = np.zeros((101,101))
	for i in range(100):
		distmat[0,i+1] = labs[i]
		distmat[i+1,0] = labs[i]
	for i in range(len(feats[:100])):
		for j in range(len(feats[:100])):
			distmat[i+1,j+1] = distCal.getDist(feats[i],feats[j])
	df = pd.DataFrame(distmat)
	df.to_csv(r'c:\temp\data.csv')
	
	print('End of Test!')
	