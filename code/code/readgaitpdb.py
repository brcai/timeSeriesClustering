import numpy as np
import matplotlib.pyplot as plt

def readPdb(seq):
	fp = open("C:\\study\\time-series\\gaitpdb\\data\\0.txt")
	feats = []
	labs = []
	for line in fp.readlines():
		tmp = line.replace('\n', '')
		tmp = tmp.split(',')
		feats.append([[float(itm)] for itm in tmp[1:]])
		labs.append(int(tmp[0]))
	fp.close()
	fp = open("C:\\study\\time-series\\gaitpdb\\data\\"+str(seq)+".txt")
	idx = 0
	for line in fp.readlines():
		tmp = line.replace('\n', '')
		tmp = tmp.split(',')
		for i in range(len(tmp) - 1):
			feats[idx][i].append(float(tmp[i+1]))
		idx += 1
	return feats, labs

if __name__ == "__main__":
	#draw feats
	while 1:
		i = input("enter sequence:")
		feats, labs = readPdb(int(i))
		idx = 0
		for j in range(len(feats)):
			dx = [feats[j][k][0] for k in range(len(feats[j]))]
			dy = [feats[j][k][1] for k in range(len(feats[j]))]
			plt.plot(dx, dy)
			plt.savefig("C:\\study\\time-series\\gaitpdb\\data\\pic\\"+str(idx)+"("+str(labs[j])+")"+".png", bbox_inches='tight', pad_inches = 0)
			plt.gcf().clear()
			idx += 1
		print("Finish drawing\n")