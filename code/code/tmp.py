import matplotlib.pyplot as plt
from rdp import rdp
import numpy as np
import re
from funcs import read

def srdp(org, eps, n):
	a = [[i, org[i]] for i in range(len(org))]
	b = [[100000,a[0]]]
	que = list([a])
	while len(que) != 0:
		c, cd, idx = getCorner(que[0])
		if cd < eps or cd == 0: que.pop(0); continue
		b.append([cd, c])
		tmp = que[0]
		que.pop(0)
		if len(tmp[0:idx]) > 2:
			que.append(tmp[0:idx])
		if len(tmp[idx:-1]) > 2:
			que.append(tmp[idx:-1])
		plt.scatter([itm[1][0] for itm in b], [itm[1][1] for itm in b])
		plt.plot(org)
		plt.show()
		plt.gcf().clear()
		break
	b.append([99999, a[-1]])
	b.sort(reverse=True)
	d = [itm[1] for itm in b[:n]]
	d.sort()
	e = [itm[1] for itm in b]
	e.sort()
	return [itm[1] for itm in e], a

def getCorner(a):
	maxdist = 0.
	idx = 0
	for i in range(1,len(a)-1):
		#(x2-x1)y+(y1-y2)x+(x1-x2)y1+(y2-y1)x1 / sqr((y1-y2)*2 + (x1-x2)*2)
		dist = getDist([a[0],a[i],a[-1]]) * distSum([itm[1] for itm in a], i)
		if dist > maxdist: idx = i; maxdist = dist
	return a[idx], maxdist, idx

def getDist(a):
    dist = 0.
    dist = abs(((a[-1][0]-a[0][0])*a[1][1] + (a[0][1]-a[-1][1])*a[1][0] + 
                    (a[0][0]-a[-1][0])*a[0][1] + (a[-1][1]-a[0][1])*a[0][0])) /\
    np.sqrt([pow(a[0][1]-a[-1][1], 2) + pow(a[0][0]-a[-1][0], 2)])
    return dist[0]

def znorm(a):
    tmp = np.array(a)
    u = np.mean(tmp)
    delta = np.std(tmp)
    b = []
    for itm in a:
        b.append((itm-u)/delta)
    return b

def getArea(l, a, r, deltal, deltar):
	dist = 0.
	x = [0,deltal,deltar]
	y = [l,a,r]
	ldist = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
	return dist

def getAngle(l, a, r, deltal, deltar):
	angle = 0.
	va = [deltal, a-l]
	vb = [deltar-deltal, r-a]
	angle = abs(180 - np.arccos(np.dot(va,vb))*180/np.pi)
	return angle

def noLower(a,b):
	if a - b< 0: return 0
	else: return a - b

def noBigger(a,b,c):
	if a+b > c: return c
	else: return a+b

def distSum(a, i):
	dist = 0.
	n = len(a)
	t = int(n/10)
	l = 0
	r = 0
	deltal = 0
	deltar = 0
	if i-t < 0:
		l = 0
		deltal = i-t
	else:
		l = i-t
		deltal = l
	if i+t > n:
		deltar = i+t - n
		r = n
	else:
		deltar = t
		r = i+t
	avgl = np.average(a[noLower(l, int(t/2)):noBigger(l, int(t/2), n)])
	avgr = np.average(a[noLower(r, int(t/2)):noBigger(r, int(t/2), n)])
	t = []
	t.append(np.average(a[noLower(i, t):i]))
	t.append(np.average(a[noLower(i, int(t/2)):noBigger(i, int(t/2), n)]))
	t.append(np.average(a[i:noBigger(i, t, n)]))
	dist = getArea(avgl, t, avgr, deltal, deltar)
	return dist



#####################################
feats, labs = read('cbf')

a = []

for i in range(1, len(feats)):
	a = feats[i]
	'''
	maxsum = 0.
	idx = 0
	for i in range(1,len(a)-1):
		sum1 = pow(getDist([[0,a[0]],[i,a[i]],[len(a),a[-1]]]), 1./1) * distSum(a, i)
		if sum1 > maxsum: maxsum = sum1; idx = i
	plt.plot(a)
	plt.scatter([idx],a[idx]+1)
	plt.show()
	'''
	b, b1 = srdp(a, 6.14685885246836292, 5)
	ttt = [itm[1] for itm in b1]
	ttt[0] = sum(ttt[:12])/12
	ttt[-1] = sum(ttt[-12:])/12
	'''
	plt.plot(a)
	plt.scatter([idx], [a[idx]+1])
	plt.show()

	print(maxsum, idx)
	print(a[idx])
	'''
print('End')