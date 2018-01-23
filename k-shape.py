import numpy as np

class kShape:
	#shape-based distance
	#len(x) == len(y)
	def SBD(x, y):
		len = pow(2, int(np.log2(2*len(x)-1))+1)
		CC = np.fft.ifft(np.fft.fft(x, len) * np.conjugate(np.fft.fft(y,len)))
		NCCc = CC/(np.linalg.norm(x)*np.linalg.norm(y))
		shift = b.index(max(NCCc)) - len(x)
		dist = 1 - max(NCCc)
		if shift > 0:
			newy = np.zeros(shift) + y[1:(len(y)-shift)]
		else:
			newy = y[(1-shift):] + np.zeros(1-shift)
		return dist, newy

	#find the proper centroids
	def ShapeExtraction():

		return

	#the main clustering algorithm
	def run():

		return
