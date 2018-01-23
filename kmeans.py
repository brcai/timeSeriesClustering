from sklearn.cluster import KMeans
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


def read():
	fp = open('C:/study/time-series/ecgfivedays/ecgfivedays')
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


if __name__ == "__main__":
	feat, label = genShift(2, 0, 0, 5)
	db = KMeans(n_clusters=2).fit(feat)
	dlabel = db.labels_
	dd = []
	for itm in dlabel:
		if itm == -1: dd.append(tmp); tmp += 1
		else: dd.append(itm)
	ars = metrics.adjusted_mutual_info_score(label, dd)
	print('kmeans ars = ' + str(ars))
	print('The end...')
