import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import csv
from sklearn import metrics
import datetime
import random
from fastCluster import read

def draw(file):
	sum = [[],[],[],[]]
	feats, labels = read(file)

	for i in range(len(labels)):
		plt.plot(feats[i])
		dir = str(labels[i])
		plt.savefig("C:/study/time-series/"+file+"/"+dir+'/'+str(i)+".png", bbox_inches='tight', pad_inches = 0)
		plt.gcf().clear()
	return

if __name__ == '__main__':
	draw('car')