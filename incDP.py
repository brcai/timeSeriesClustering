import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn import metrics
from timeSeriesGen import genShift

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

class dpOrig:
	#calculate p and delta
	def calcParams(self, dataVecs, dc, kl):
		wordParam = {}
        #calculate p as param[0], neighbours as param[1]
		maxp = 0
		maxData = 0
		maxDist = 0.0
		for i in range(0, len(dataVecs)):
			cnt = 0
			neighbours = []
			for j in range(0, len(dataVecs)):
				if i!=j:
					tmp = euclidean(dataVecs[i], dataVecs[j])
					tmpDist = 0.
					if tmp < dc: 
						#normal regularization
						if kl == 'nl':
							cnt += 1-(tmp**2/dc**2); neighbours.append(j)
						#gaussian kernel
						elif kl == 'gs':
							cnt += np.exp(-tmp**2/dc**2); neighbours.append(j)
						#normal distance
						elif kl == 'non': 
							cnt += 1; neighbours.append(j)
					if tmp > maxDist: maxDist = tmp
			wordParam[i] = [cnt, neighbours]
			if maxp < cnt: maxp = cnt; maxData = i
		#calculate delta as param[2], nearest higher density point j
		for i in range(0, len(dataVecs)):
			minDelta = maxDist
			affiliate = -1
			for j in range(0, len(dataVecs)):
				if wordParam[j][0] > wordParam[i][0]: 
					#euclidean distance
					tmp = np.linalg.norm(np.array(dataVecs[i]) - np.array(dataVecs[j]))
					if minDelta > tmp: 
						minDelta = tmp
						affiliate = j
			wordParam[i].extend([minDelta, affiliate])
		return wordParam

	def assignCluster(self, wordParams, centers, dc, lasso):
		pRank = [[wordParams[word][0], word] for word in wordParams]
		pRank.sort(reverse = True)
		wordCluster = {word:-1 for word in wordParams}
		id = 0
		centre2cluster = {}
		self.plotDgraph(wordParams, centers)
		for p in pRank:
			if wordCluster[p[1]] == -1: 
				if p[1] in centers: wordCluster[p[1]] = id; centre2cluster[p[1]] = id; id += 1
				else: 
					if wordParams[p[1]][3] == -1: 
						#print('error, increase dc and try again....\n') 
						return wordCluster, False
					wordCluster[p[1]] = wordCluster[wordParams[p[1]][3]]
		return wordCluster, True

	def read(self, dir):
		fp = open('C:/study/clustering dataset/' + dir)
		dx = []
		dy = []
		id = []
		num = 0
		clusters = []
		for line in fp.readlines():
			raw = re.split('[ |\t]', line)
			tmp = [itm for itm in raw if itm != '']
			arr = [float(itm.replace('\n', '')) for itm in tmp]
			dx.append(arr[0])
			dy.append(arr[1])
			if len(arr) == 3: 
				id.append(int(arr[2]))
				if arr[2] not in clusters:
					clusters.append(arr[2])
					num += 1
		return dx, dy, id, num

	def calcMaxDist(self, dataVecs, dc):
		rawDists = set()
		for i in range(len(dataVecs)):
			for j in range(i + 1, len(dataVecs)):
				dist = euclidean(dataVecs[i], dataVecs[j])
				rawDists.add(dist)
		dists = list(rawDists)
		dists.sort()
		return dists

	def getCenters(self, wordParams, num, dc):
		centers = []
		pRank = []
		for itm in wordParams:
			pRank.append([wordParams[itm][2] * wordParams[itm][0], itm])
		pRank.sort(reverse = True)
		if num > len(pRank): return centers
		centers = [itm[1] for itm in pRank[0:num]]
		return centers

	def plotDgraph(self, wordParams, centers):
		dx = [wordParams[itm][0] for itm in wordParams]
		dy = [wordParams[itm][2] for itm in wordParams]
		dpcolorMap3 = ['k' for i in range(len(dx))]
		cx = []
		cy = []
		for center in centers:
			cx.append(dx[center])
			cy.append(dy[center])
		ident = 0	
		dddd
		for center in centers:
			dx.pop(center)
			dy.pop(center)
			ident += 1
		
		plt.scatter(dx, dy, c=dpcolorMap3, marker='.', s=300, alpha=0.5)
		dpcolorMap1 = ['r']
		plt.scatter(cx, cy, c=dpcolorMap1, marker='.', s=1000,edgecolor='k')
		plt.xlabel(r'$\rho$', fontsize=25)
		plt.ylabel(r'$\delta$', fontsize=25)
		
		plt.show()
		return


	def plotClusterRes(self, wordCluster, dx, dy, lasso, centers, dir):
		colors = cm.rainbow(np.linspace(0, 1, lasso + 1))
		dpcolorMap2 = [colors[wordCluster[itm]] for itm in range(len(dx))]
		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=150, alpha=0.8, zorder=1, edgecolor='k')

		cx = []
		cy = []
		for itm in centers:
			cx.append(dx[itm])
			cy.append(dy[itm])
		#plt.scatter(cx, cy, c='r',marker='+', s=400, zorder=2)
		plt.text(14.5, 1.0, r'$NMI=$'+str(0.5038), fontsize=15, color='black')
		#plt.xlabel('X', fontsize=25)
		#plt.ylabel('Y', fontsize=25)
		plt.xticks([])
		plt.yticks([])
		plt.axis('off')
		#plt.savefig("C:\\study\\8data\\Compound_org.png", bbox_inches='tight', pad_inches = 0)
		plt.savefig("C:\\study\\8data\\"+dir.split('.')[0]+"_dp.png", bbox_inches='tight', pad_inches = 0)
		plt.show()
		return

	def run(self, dir, dc, lasso, kl):
		feats, trueLabels = genShift(2, 0, 0, 1)
		dataVecs = feats
		dists = self.calcMaxDist(dataVecs, dc)
		realDc = dists[-1]*dc
		#print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(dataVecs, realDc, kl)
		centers = self.getCenters(wordParams, lasso, realDc)
		if len(centers) == 0: return wordCluster, label, arsTmp, amiTmp
		wordCluster, valid = self.assignCluster(wordParams, centers, realDc, lasso)
		label = [wordCluster[i] for i in range(len(dataVecs))]
		amiTmp = metrics.adjusted_mutual_info_score(trueLabels, label)
		print(amiTmp)
		return 

if __name__ == "__main__":
	inst = dpOrig()
	for i in frange(0.01, 0.8, 0.05):
		inst.run('jain.txt', 0.06, 2, 'nl')
	print('The end...')
