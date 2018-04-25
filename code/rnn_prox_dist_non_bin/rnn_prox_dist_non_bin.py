#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

import pickle

###### Parameters

'''
## External Grammar

Markov_Gramm = read_ppm("../misc_data/grammar_mat_simple.ppm")

letter_node_ind,letters,inp_node_ind = read_letter_map("../misc_data/letter_mapping_simple.csv")

N_letter_nodes = inp_node_ind.shape[0]
N_ext = inp_node_ind.max() + 1 # number of input nodes (features, letters...)

p_mat_node_letter = np.zeros((N_ext,N_letter_nodes))

for k in range(N_letter_nodes):
	p_mat_node_letter[inp_node_ind[k],k] = 1.
##
'''

## Network
N_e = 300 # number of excitatory neurons

sigma_w_init = 1.
mu_w_init = 0.

mu_learn = 0.02


act_fix = 0.25
a_hebb = -np.log(1./act_fix - 1.)
##

## Simulation

n_t = 15000
n_t_skip = 50
n_t_rec = int(n_t/n_t_skip)
##

param_dict = {	"N_e":N_e,
				"sigma_w_init":sigma_w_init,
				"mu_w_init":mu_w_init,
				"mu_learn":mu_learn,
				"act_fix":act_fix,
				"a_hebb":a_hebb,
				"n_t":n_t,
				"n_t_skip":n_t_skip,
				"n_t_rec":n_t_rec}




def h(x_post_pot,x_pre,a):

	f = (x_post_pot*np.tanh(a)/a - np.tanh(x_post_pot))/a**3

	return np.outer(f,x_pre)


def act(x,gain):

	return (np.tanh(gain*x/2.)+1.)/2.


def main(x_e = np.random.rand((N_e)), W=np.random.normal(mu_w_init, sigma_w_init, (N_e,N_e)), path_x="./data/x_e_rec",path_W="./data/W_rec",path_I="./data/I_ee_rec",path_p="./data/parameters.p"):

	W = np.random.normal(mu_w_init, sigma_w_init, (N_e,N_e))
	W[range(N_e),range(N_e)] = 0.

	x_e = np.random.rand((N_e))

	## recording

	x_e_rec = np.ndarray((n_t_rec,N_e))

	W_rec = np.ndarray((n_t_rec,N_e,N_e))

	I_ee_rec = np.ndarray((n_t_rec,N_e))

	##

	for t in tqdm(range(n_t)):

		I_ee = np.dot(W,x_e)

		x_e_new = act(I_ee,2.)
		#pdb.set_trace()
		W += mu_learn*h(I_ee,x_e,a_hebb)
		W[range(N_e),range(N_e)] = 0.

		if t%n_t_skip == 0:
			x_e_rec[int(t/n_t_skip),:] = x_e_new
			W_rec[int(t/n_t_skip),:,:] = W
			I_ee_rec[int(t/n_t_skip),:] = I_ee
		
		x_e = x_e_new

	#plt.plot(W_rec[:,0,:])
	#plt.show()

	#plt.plot(x_e_rec)
	#plt.show()

	np.save(path_x,x_e_rec)
	np.save(path_W,W_rec)
	np.save(path_I,I_ee_rec)
	
	with open(path_p,"wb") as file:
		pickle.dump(param_dict,file)

	#pdb.set_trace()


if __name__ == "__main__":
	x = np.random.rand((N_e))
	W = np.random.normal(mu_w_init, sigma_w_init, (N_e,N_e))

	main(x_e = x, W=W, path_x="./data/x_e_rec_1",path_W="./data/W_rec_1",path_I="./data/I_ee_rec_1",path_p="./data/parameters_1.p")
	
	x = x+np.random.rand((N_e))*0.00001
	W = W+np.random.normal(mu_w_init, sigma_w_init, (N_e,N_e))*0.00001

	main(x_e = x, W=W, path_x="./data/x_e_rec_2",path_W="./data/W_rec_2",path_I="./data/I_ee_rec_2",path_p="./data/parameters_2.p")
	