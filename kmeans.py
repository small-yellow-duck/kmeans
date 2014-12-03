import random
import numpy as np



def do_kmeans(data, num_clusters, n_iter=10):

	best_score = np.inf
	
	
	for i in range(6):
		cluster_centres, labels, new_score = kmeans_new_init(data, num_clusters, n_iter=10)
		print new_score
		if i == 0 or new_score < best_score:
			best_score = new_score
			best_labels = labels
			best_cluster_centres = cluster_centres
			
	print 'best score/num_clusters**2 ', best_score/(num_clusters**3)		
	return best_cluster_centres, best_labels		


def kmeans_new_init(data, num_clusters, n_iter=10):

	radius = np.zeros(num_clusters)
	
	data = np.array(data, dtype=float)
	
	pts = data.shape[0]
	if num_clusters > pts:
		num_clusters = pts
	
	init_ind = range(pts)
	random.shuffle(init_ind)
	init_ind = init_ind[0:num_clusters]
	
	cluster_centres = data[init_ind, :]
	
	
# 	print cluster_centres.shape
# 	print np.tile(cluster_centres[0,:], pts).reshape((pts,-1))[0:2, :]
	
	i = 0
	old_score = 0.0
	new_score = np.inf
	
	while i < n_iter and ((new_score < 0.999*old_score) or i==0):
		old_score = 1.0*new_score
		dist = np.zeros((pts, num_clusters))
		for c in range(cluster_centres.shape[0]):
			dist[:, c] = np.sum( np.abs(data - np.tile(cluster_centres[c,:], pts).reshape((pts,-1)) )**2, axis=1)
			#radius[c] = np.sort(dist[:,c])[int(0.9*pts)]
		
		labels = np.argmin(dist, axis=1)
		new_score = np.sum(np.min(dist, axis=1))
		#print '---- ', new_score
		
		for c in range(cluster_centres.shape[0]):
			if data[labels==c,:].shape[0] > 0:
				cluster_centres[c,:] = np.mean(data[labels==c,:], axis=0)
			else:	
				cluster_centres[c,:] = data[random.randint(0,pts),:]
		
		i = i+1		
	
	#print '-----'
	#print np.min(radius)/np.max(radius)
	
	return cluster_centres, labels, new_score