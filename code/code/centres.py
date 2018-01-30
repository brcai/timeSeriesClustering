import numpy as np
from random import random
from timeSeriesGen import genShift

FLOAT_MAX = 1e100 # 设置一个较大的值作为初始化的最小的距离

def distance(vecA, vecB):
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]

def getDense(feats, idx):
	dense = 0.
	for dx, itm in enumerate(feats):
		if idx == dx: continue
		dense += 1/distance(np.mat(itm), np.mat(feats[idx]))
	return dense

def nearest(point, cluster_centers):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

def get_centroids(points, k):
	points = np.mat(points)
	m, n = np.shape(points)
	cluster_centers = np.mat(np.zeros((k , n)))
    # 1、随机选择一个样本点为第一个聚类中心
	index = np.random.randint(0, m)
	cluster_centers[0, ] = np.copy(points[index, ])
    # 2、初始化一个距离的序列
	d = [0.0 for _ in range(m)]

	for i in range(1, k):
		sum_all = 0
		for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
			d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
			sum_all += d[j]
        # 5、取得sum_all之间的随机值
		sum_all *= random()
        # 6、获得距离最远的样本点作为聚类中心点
		for j, di in enumerate(d):
			sum_all -= di
			if sum_all > 0: continue
			cluster_centers[i] = np.copy(points[j, ])
			break
	centres = cluster_centers.tolist()
	return centres

if __name__ == '__main__':
	feats, trueLabels = genShift(2, 0, 0, 1)
	centres = get_centroids(feats, 2)
	print('end of test!!')