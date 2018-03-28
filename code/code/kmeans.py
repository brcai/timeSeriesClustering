from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from funcs import *
from sklearn.datasets.samples_generator import make_blobs

class Kmeans:
	def __init__(self, k, iter):
		X, y_true = make_blobs(n_samples=300, centers=4,
		cluster_std=0.60, random_state=0)
		self.k = k
		self.iter = iter
		#feats, labs = read('cbf')
		feats, labs = X, y_true
		self.feats = feats
		self.labs = labs
		tmp = np.random.rand(self.k,1)
		self.centres = []
		for itm in tmp:
			val = int(itm[0]*len(feats))
			if val in self.centres: self.centres.append(val+1)
			else: self.centres.append(val)
		self.clusters = []

	def getCentre(self, feat):
		mindist = 10000000.
		idx = -1
		for centre in range(len(self.centres)):
			if euclidean(feat, self.feats[centre]) < mindist: mindist = euclidean(feat, self.feats[centre]); idx = centre
		return idx

	def fit(self):
		iter = 0
		centre = self.centres
		while iter < self.iter or centre != self.centres:
			centre = self.centres
			clusters = [[] for i in range(self.k)]
			for feat in range(len(self.feats)):
				clusters[self.getCentre(self.feats[feat])].append(feat)
			for centre in range(len(self.centres)):
				newc = []
				for i in range(len(self.feats[feat])):
					sum = 0.
					for j in range(len(self.feats)):
						sum += self.feats[j][i]
					avg = sum / len(self.feats)
				newc.append(avg)
			self.centres = newc
		for key in clusters:
			for itm in clusters[key]:
				self.clusters[itm] = key

if __name__ == "__main__":
	inst = Kmeans(4, 10)
	inst.fit()
	print('Kmeans (dtwdist):')
	feats, labs = read('cbf')



	print('The end...')
