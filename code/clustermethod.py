#!/usr/bin/env python3


import numpy as np
import pdb

# A "fork" in the cluster tree is saved as:
# [[list of elements of fork (two or one)],distance between elements,position_of_fork_itself]

## Hamming Distance
def hamm_d(x1,x2):
	return np.abs(x1-x2).sum()
##

## Squared Eucl. Distance
def squ_d(x1,x2):
	return ((x1-x2)**2.).sum()
##

def midpoint_eucl(x1,x2):
	return (x1+x2)/2.




def clust_tree(obj_list,metric,midpoint):



	n_obj = len(obj_list)

	fork_list = []

	for k in range(n_obj):
		fork_list.append({"elements":[k],"dist":0.,"pos":obj_list[k]})

	n = 0
	while len(fork_list) > 1:

		for k in range(len(fork_list)):

			for l in range(k):
				if k==1 and l==0:
					closest_pair = [k,l]
					d_closest = metric(fork_list[k]["pos"],fork_list[l]["pos"])
				else:
					d = metric(fork_list[k]["pos"],fork_list[l]["pos"])
					if d < d_closest:
						closest_pair = [k,l]
						d_closest = d
		new_fork = {"elements":[fork_list[closest_pair[0]],fork_list[closest_pair[1]]],
					"dist":d_closest,
					"pos":midpoint(fork_list[closest_pair[0]]["pos"],fork_list[closest_pair[1]]["pos"])}
		
		fork_list[closest_pair[0]] = new_fork
		del fork_list[closest_pair[1]]
		
		n+=1

	return fork_list


obj = []

for k in range(10):
	obj.append(np.random.normal(0.,.1,(2)))
	obj.append(np.random.normal(2.,.1,(2)))

f_list = clust_tree(obj,squ_d,midpoint_eucl)

pdb.set_trace()