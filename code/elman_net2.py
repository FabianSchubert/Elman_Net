#!/usr/bin/python3

import sys
sys.path.append("../../../custom_modules/")
from plot_settings_mpl import *

import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))


def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage


def s(x):
	return np.tanh(x)
	#return (np.tanh(x/2.)+1.)/2.

def ds(x):
	return 1.-s(x)**2
	#return s(x)*(1.-s(x))


def update_layers(I,h,c,W_h,W_o,n_i,n_h,n_o):

	I_h = np.ones((n_i+n_h+1))
	I_o = np.ones((n_h+1))
	
	I_h[:n_i] = I
	I_h[n_i:n_i+n_h] = c

	I_o[:n_h] = h

	h_new = s(np.dot(W_h,I_h))
	c_new = h[:]
	o_new = s(np.dot(W_o,I_o))

	return (h_new,c_new,o_new)

def update_weights(I,c,h,o,f,W_h,W_o,n_i,n_h,n_o,mu_learn,mu_learn_out):

	dh = ds(h)
	do = ds(o)

	I_h = np.ones((n_i+n_h+1))
	I_o = np.ones((n_h+1))
	
	I_h[:n_i] = I
	I_h[n_i:n_i+n_h] = c

	I_o[:n_h] = h

	dW_h = np.zeros((n_h,n_i+n_h+1))
	dW_o = np.zeros((n_o,n_h + 1))

	#pdb.set_trace()

	dW_o = -mu_learn_out * np.outer(do*(o-f),I_o)
	dE_h = np.dot(W_o.T[:n_h,:],do*(o-f))

	dW_h = -mu_learn * np.outer(dE_h*dh,I_h)

	return dW_h, dW_o
	
	
	

def main():

	n_i = 1
	n_h = 50
	n_o = 1

	n_t = 1000000

	mu_learn = np.ones((n_h,n_i+n_h+1))*.005
	mu_learn[:,-1] = .0001
	mu_learn_out = np.ones((n_o,n_h + 1))*.005
	mu_learn_out[:,-1] = .0001

	momentum = 0.

	I = np.ndarray((n_t,n_i))
	
	for k in range(n_t):
		if k%3 == 2:
			I[k,0] = 1.*(I[k-1,0] != I[k-2,0])
		else:
			I[k,0] = int(np.random.rand()+.5)
	#I[:,0] = np.sin(np.array(range(n_t))*0.125*2.*np.pi)*0.125 + np.sin(np.array(range(n_t))*0.0625*2.*np.pi)*0.125
	I = (I-.5)*.5
	#I = I*0.8 + 0.1
	#pdb.set_trace()


	#f = np.ndarray((n_t,n_o))
	#f[:,0] = np.sin(np.linspace(0.,50*2.*np.pi,n_t))*0.25+.5
	#f[:,1] = f[:,0]

	h = np.zeros(n_h)
	c = np.zeros(n_h)
	o = np.zeros(n_o)

	h_rec_bpp = np.zeros((2,n_h))

	W_h = np.random.rand(n_h,n_i+n_h+1)*0.2 - .1 #external input + input from context unit + 1 input for "bias weight"

	W_o = np.ones((n_o,n_h+1))#np.random.rand(n_o,n_h + 1)*1. - .5 #input from hidden unit + 1 bias weight
	W_o[:,-1] = 0.

	dW_h = np.zeros((n_h,n_i+n_h+1))
	dW_o = np.zeros((n_o,n_h + 1))

	h_rec = np.ndarray((n_t,n_h))
	o_rec = np.ndarray((n_t,n_o))

	E_rec = np.ndarray((n_t))
	
	#W_h_rec = np.ndarray((n_t,n_h,n_i+n_h+1))
	#W_o_rec = np.ndarray((n_t,n_o,n_h + 1))

	for t in tqdm(range(n_t)):
		h_rec_bpp[0,:] = h_rec_bpp[-1,:]
		h_rec_bpp[-1,:] = h[:]

		h,c,o = update_layers(I[t,:],h,c,W_h,W_o,n_i,n_h,n_o)
		
		if t<n_t*0.75:# and (t+1)%3 == 2:
			#I,c,h,o,f,W_h,W_o,n_i,n_h,n_o,mu_learn,mu_learn_out
			dW_h_old = np.array(dW_h)
			dW_o_old = np.array(dW_o)
			dW_h,dW_o = update_weights(I[t-1,:],h_rec_bpp[-2,:],h_rec_bpp[-1,:],o,I[t+1,:],W_h,W_o,n_i,n_h,n_o,mu_learn,mu_learn_out)
			W_h += dW_h_old * momentum + (1.-momentum) * dW_h#*(1.-momentum) + momentum*dW_h
			W_o += dW_o_old * momentum + (1.-momentum) * dW_o#*(1.-momentum) + momentum*dW_o
			#dW_h = dW_h_new[:,:]
			#dW_o = dW_o_new[:,:]

		h_rec[t,:] = h
		o_rec[t,:] = o
		
		#W_h_rec[t,:,:] = W_h
		#W_o_rec[t,:,:] = W_o

		if t<n_t-1:
			E_rec[t] = np.linalg.norm(o-I[t+1])**2

	corr_mat = np.corrcoef(h_rec.T)
	dist_mat = 1.- 0.5*(corr_mat + corr_mat.T)
	dist_mat[range(n_h),range(n_h)] = 0.
	pdb.set_trace()
	cluster = compute_serial_matrix(dist_mat,method="complete")



	fig1, ax1 = plt.subplots(2,1)
	ax1[0].plot(I[1:],label='Input')
	ax1[0].plot(o_rec[:,0],'--',c='k',label="Predicted Output")
	
	ax1[0].set_xlim([n_t-100,n_t])

	ax1[0].legend()

	ax1[1].plot(I[1:])
	ax1[1].plot(o_rec,'--',c='k')
	
	ax1[1].set_xlim([0,1000])

	ax1[1].set_xlabel("Time Step")

	#plt.show()

	#pdb.set_trace()
	'''
	fig2 = plt.figure()
	for k in range(n_h):
		plt.plot(W_h_rec[:,k,:],c='k',alpha=0.8)
	plt.xlabel("Time Step")
	plt.ylabel("W_h")
	'''

	#plt.show()
	'''
	fig3 = plt.figure()
	for k in range(n_o):
		plt.plot(W_o_rec[:,k,:],c='k',alpha=0.8)
	plt.xlabel("Time Step")
	plt.ylabel("W_o")
	'''

	plt.show()
	pdb.set_trace()

if __name__ == "__main__":
	main()
