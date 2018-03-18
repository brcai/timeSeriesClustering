from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from funcs import *

def brdp(a, eps):
	p = []
	l = int(len(a) / 10)

	s = sum(a[:l]) / l
	e = sum(a[-l:]) / 1
	
	sigmay2 = sum([pow(itm,2) for itm in a[1:-1]])
	sigmag2 = 0.
	sigmaf2 = 0.
	sigmayg = 0.
	sigmayf = 0.

	maxv = 0.

	for i in range(1, len(a)-1):
		

	return p


if __name__ == "__main__":
	print('Kmeans (dtwdist):')
	feats, labs = read('cbf')



	print('The end...')
