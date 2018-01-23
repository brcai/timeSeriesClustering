import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import csv
from sklearn import metrics

def readEcgFiveDays():
	fp = open('C:/study/time-series/ecgfivedays/ecgfivedays.txt')
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

def readEcg200():
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


if __name__ == "__main__":
	feat, label = readEcg200()
	i = 0
	for idx, ll in enumerate(label):
		if ll == 2:
			plt.plot(feat[idx])	
		#elif ll == 2:
			#plt.savefig("C:/study/time-series/ecgfivedays/2/"+"sum.png", bbox_inches='tight', pad_inches = 0)
		#plt.gcf().clear()
		#i += 1
	plt.savefig("C:/study/time-series/ecg200/2/"+"sum.png", bbox_inches='tight', pad_inches = 0)
	print("end!!")