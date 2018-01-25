import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import csv
from sklearn import metrics
import datetime
import random
from sklearn.decomposition import PCA

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

def genShift(classNum, length, dataNum, step):
	base1 = [1.2807,1.5708,1.9153,2.3362,2.7351,2.9284,2.8112,2.4533,2.0379,1.6965,1.4205,1.1254,0.75797,0.35345,-0.002452,-0.23449,-0.31305,-0.27555,
	   -0.20967,-0.21105,-0.31069,-0.43958,-0.49433,-0.4358,-0.33923,-0.31363,-0.38318,-0.47433,-0.4964,-0.4357,-0.35011,-0.28571,-0.25934,-0.27502,
	   -0.33756,-0.41314,-0.4167,-0.31361,-0.20237,-0.2565,-0.50839,-0.76383,-0.78701,-0.57426,-0.39353,-0.48503,-0.80027,-1.0652,-1.0917,-0.96191,
	   -0.87657,-0.9195,-1.0095,-1.0647,-1.1147,-1.2139,-1.3277,-1.3543,-1.2686,-1.1697,-1.1631,-1.2366,-1.2731,-1.1908,-1.0229,-0.86577,-0.78111,
	   -0.7522,-0.72919,-0.66635,-0.5354,-0.33547,-0.10215,0.087808,0.17387,0.16455,0.13113,0.14911,0.23193,0.35334,0.49033,0.62967,0.7392,0.75483,
	   0.66171,0.5438,0.53068,0.66927,0.85229,0.92928,0.85299,0.71977,0.65628,0.68352,0.7293,0.72617]
	base2 = [-0.19441,-1.1007,-1.3216,-0.81488,-0.34072,-0.38421,-0.6235,-0.64221,-0.55257,-0.58375,-0.67414,-0.70857,-0.7342,-0.76711,-0.74069,
	   -0.69701,-0.73889,-0.84595,-0.90055,-0.88965,-0.88328,-0.92933,-1.0131,-1.1151,-1.2142,-1.2802,-1.2905,-1.2989,-1.4081,-1.5917,-1.6842,
	   -1.6191,-1.5579,-1.6148,-1.6867,-1.6488,-1.5526,-1.4599,-1.3188,-1.1241,-0.98348,-0.91711,-0.76622,-0.4358,-0.062335,0.14746,0.20885,
	   0.29907,0.4815,0.61784,0.65201,0.70057,0.82487,0.92154,0.97288,1.0792,1.1797,1.1044,0.92291,0.86882,0.92704,0.92672,0.88343,0.89989,0.89992,
	   0.84429,0.87548,0.98458,0.91121,0.64469,0.54624,0.74881,0.9048,0.76475,0.56709,0.63115,0.91369,1.16,1.25,1.2581,1.2938,1.3502,1.3011,1.0883,
	   0.86683,0.81278,0.85455,0.81905,0.76498,0.8761,1.0859,1.1447,1.022,0.92983,0.96388,1.0142]
	num = dataNum / classNum
	feats1 = [normlize(base1)]
	feats2 = [normlize(base2)]
	tmp1 = normlize(base1).copy()
	tmp2 = normlize(base2).copy()
	for i in range(49):
		tmp1 = tmp1[step:] + tmp1[:step]
		feats1.append(tmp1)
		tmp2 = tmp2[step:] + tmp2[:step]
		feats2.append(tmp2)
	feats = feats1[3:10] + feats1[:3] + feats2[:40] + feats1[10:] + feats2[40:]
	labs = [0 for i in range(10)] + [1 for i in range(40)] + [0 for i in range(40)] + [1 for i in range(10)]
	return feats, labs

def plot(feats):
	for idx, itm in enumerate(feats):
		print(idx)
		plt.plot(itm)
		plt.savefig("C:/study/time-series/synthetic/1/"+str(idx)+".png", bbox_inches='tight', pad_inches = 0)
		plt.gcf().clear()
	return

if __name__ == "__main__":
	print("start gen time series:")
	feats, labs = genShift(2, 0, 0, 1)
	plot(feats)
	print("end of test!!")