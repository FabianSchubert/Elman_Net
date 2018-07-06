#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

from grammar_data_reader import *

import pickle

import ast

import sys




def h(X, y_post, y, y_mean, a):
	f = a*(2.+a*X*(1.-2*y_post))*(2*y_post-1.+2.*a*X*(1.-2.*y_post)*y_post)
	return np.outer(f,y-y_mean)

def dgain(X,y_post,th,gain,l1,l2):
	std = 1./(-l2*2.)**.5
	mean = -l1/(2.*l2)
	theta = (1-2.*y_post)+y_post*(1.-y_post)*(l1+2.*l2*y_post)
	return 1./gain + (X-th)*theta
	#return std**2 - (y_post - mean)**2

def dthresh(y_post,gain,l1,l2):
	mean = -l1/(2.*l2)
	theta = (1-2.*y_post)+y_post*(1.-y_post)*(l1+2.*l2*y_post)
	return -gain*theta
	#return y_post - mean
	

def h_approx(x_post_pot,x_pre,x_pre_mean,a):

	#f = (x_post_pot*np.tanh(a)/a - np.tanh(x_post_pot))/a**3
	#f = (x_post_pot**2)*(a-x_post_pot)
	#f = (-x_post_pot*(x_post_pot-a)*(x_post_pot+a))/a**2
	l = 0.958/a

	f = -x_post_pot + np.tanh(2*l*x_post_pot)/l 

	#return np.outer(f,x_pre-x_pre_mean)# - np.outer(x_pre,x_post)
	return np.outer(f,x_pre)# - np.outer(x_pre,x_post)

def act(x,gain,th):

	return (np.tanh(gain*(x-th)/2.)+1.)/2.


def main(x_e, W, gain_neuron, thresh_neuron, params_path = "./sim_parameters/no_input.csv",gramm_path = "../../misc_data/grammar_mat_simple.ppm", letter_map_path = "../../misc_data/letter_mapping_simple.csv" , path="./data/sim_data"):


	###### Parameters

	'''
	## External Grammar

	Markov_Gramm = read_ppm(gramm_path)

	letter_node_ind,letters,inp_node_ind = read_letter_map(letter_map_path)

	N_letter_nodes = inp_node_ind.shape[0]
	N_ext = inp_node_ind.max() + 1 # number of input nodes (features, letters...)

	p_mat_node_letter = np.zeros((N_ext,N_letter_nodes))

	for k in range(N_letter_nodes):
		p_mat_node_letter[inp_node_ind[k],k] = 1.
	##
	'''



	N_ext = 9
	N_letter_nodes = 9

	param_dict = {}

	with open(params_path,"r") as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			param_dict[row[0]] = ast.literal_eval(row[1])

	param_dict["N_letter_nodes"] = N_letter_nodes
	param_dict["N_ext"] = N_ext

	## Network
	N_e = int(param_dict["N_e"]) # number of excitatory neurons

	sigma_w_init = float(param_dict["sigma_w_init"])
	mu_w_init = float(param_dict["mu_w_init"])

	mu_learn_ext = float(param_dict["mu_learn_ext"])
	mu_learn_int = float(param_dict["mu_learn_int"])
	mu_learn_gain = float(param_dict["mu_learn_gain"])
	mu_learn_thresh = float(param_dict["mu_learn_thresh"])

	mean_act_target = float(param_dict["mean_act_target"])
	std_act_target = float(param_dict["std_act_target"])
	# Parameters of target function exp(l1*x + l2*x**2):
	l1 = mean_act_target/std_act_target**2
	l2 = -1./(2.*std_act_target**2)

	mu_act_av = float(param_dict["mu_act_av"])


	
	#act_fix = float(param_dict["act_fix"])
	#a_hebb = -np.log(1./act_fix - 1.)/gain_neuron

	#print(a_hebb)

	w_mean_pre_ext_input = float(param_dict["w_mean_pre_ext_input"])
	w_std_pre_ext_input = float(param_dict["w_std_pre_ext_input"])

	store_w= param_dict["store_w"]
	store_w_ext = param_dict["store_w_ext"]

	ext_plast = param_dict["ext_plast"]
	int_plast = param_dict["int_plast"]

	int_gain_adapt = param_dict["int_gain_adapt"]
	int_thresh_adapt = param_dict["int_thresh_adapt"]

	staging = param_dict["staging"]



	##

	## Simulation

	n_t = int(float(param_dict["n_t"]))
	n_t_rec = int(float(param_dict["n_t_rec"]))
	n_t_skip = int(n_t/n_t_rec)#500
	n_t_rec = int(n_t/n_t_skip)
	##

	

	#print ("Initialize Ext->I")
	if w_std_pre_ext_input == 0:
		W_eext = np.zeros((N_e,N_ext))
	else:
		W_eext = np.random.normal(w_mean_pre_ext_input,w_std_pre_ext_input,(N_e,N_ext))

	##Initialize Input
	x_letter_node = np.zeros((N_letter_nodes))
	x_letter_node[0] = 1.

	x_ext_mean = np.zeros((N_letter_nodes))

	std_ext = np.ones(N_ext)*0.
	#std_ext[0] = 5.
	'''
	M_mod_rand_inp = np.eye(N_ext)
	M_mod_rand_inp[:,0] = np.array([1.,1.,0.,0.,0.,0.,0.,0.,0.])
	M_mod_rand_inp[:,0] /= np.linalg.norm(M_mod_rand_inp[:,0])
	M_mod_rand_inp[:,1] = np.array([-1.,1.,0.,0.,0.,0.,0.,0.,0.])
	M_mod_rand_inp[:,1] /= np.linalg.norm(M_mod_rand_inp[:,1])
	'''

	#std_ext[1] = 10.
	#std_ext[2] = 2.5

	##





	#W = np.random.normal(mu_w_init, sigma_w_init, (N_e,N_e))
	W=np.array(W)
	W[range(N_e),range(N_e)] = 0.
	x_e = np.array(x_e)
	x_e_mean = x_e[:]

	gain_neuron = np.array(gain_neuron)
	thresh_neuron = np.array(thresh_neuron)

	#x_e = np.random.rand((N_e))

	## recording

	x_e_rec = np.ndarray((n_t_rec,N_e))
	x_e_mean_rec = np.ndarray((n_t_rec,N_e))

	x_ext_rec = np.ndarray((n_t_rec,N_ext))
	x_ext_mean_rec = np.ndarray((n_t_rec,N_ext))

	if store_w:
		W_rec = np.ndarray((n_t_rec,N_e,N_e))
	if store_w_ext:
		W_eext_rec = np.ndarray((n_t_rec,N_e,N_ext))

	gain_rec = np.ndarray((n_t_rec,N_e))
	thresh_rec = np.ndarray((n_t_rec,N_e))

	I_ee_rec = np.ndarray((n_t_rec,N_e))
	I_eext_rec = np.ndarray((n_t_rec,N_e))

	##

	for t in tqdm(range(n_t)):


		###CUSTOM_COMMANDS_FLAG###
		
		''' 
		## Generate Markov Grammar Input
		p_next = np.dot(Markov_Gramm,x_letter_node)
		next_letter_node = np.random.choice(N_letter_nodes,p=p_next)
		x_letter_node = np.zeros((N_letter_nodes))
		x_letter_node[next_letter_node] = 1.
		x_ext = np.zeros(N_ext)
		x_ext[inp_node_ind[next_letter_node]] = 1.
		input_letter = letters[next_letter_node]
		##
		'''

		## Generate Random Input

		x_ext = (np.random.rand(N_ext)-0.5)*std_ext*2.*3.**.5# (0.,std_ext,(N_ext))
		#x_ext = np.dot(M_mod_rand_inp,x_ext)
		##


		I_ee = np.dot(W,x_e)# + np.dot(W_eext,x_ext)
		I_eext = np.dot(W_eext,x_ext)
		I = I_ee + I_eext

		x_e_new = act(I,gain_neuron,thresh_neuron)

		x_e_mean += mu_act_av*(x_e_new - x_e_mean)
		x_ext_mean += mu_act_av*(x_ext - x_ext_mean)

		#pdb.set_trace()

		if int_plast:
			W += mu_learn_int*h(I,x_e_new,x_e,x_e_mean,gain_neuron)
			W[range(N_e),range(N_e)] = 0.
		if int_gain_adapt:
			gain_neuron += mu_learn_gain*dgain(I,x_e_new,thresh_neuron,gain_neuron,l1,l2)
			gain_neuron = np.maximum(gain_neuron,0.01)
		if int_thresh_adapt:
			thresh_neuron += mu_learn_thresh*dthresh(x_e_new,gain_neuron,l1,l2)	
		if ext_plast:
			W_eext += mu_learn_ext*h(I,x_e_new,x_ext,x_ext_mean,gain_neuron)

		
		

		#W = (W.T / W.std(axis=1)).T
		

		
		if t%n_t_skip == 0:
			x_e_rec[int(t/n_t_skip),:] = x_e_new
			x_e_mean_rec[int(t/n_t_skip),:] = x_e_mean
			x_ext_rec[int(t/n_t_skip),:] = x_ext
			x_ext_mean_rec[int(t/n_t_skip),:] = x_ext_mean
			if store_w:
				W_rec[int(t/n_t_skip),:,:] = W
			if store_w_ext:
				W_eext_rec[int(t/n_t_skip),:,:] = W_eext
			gain_rec[int(t/n_t_skip),:] = gain_neuron
			thresh_rec[int(t/n_t_skip),:] = thresh_neuron
			I_ee_rec[int(t/n_t_skip),:] = I_ee
			I_eext_rec[int(t/n_t_skip),:] = I_eext
		
		'''
		if t >= n_t-n_t_rec:# t%n_t_skip == 0:
			x_e_rec[t+n_t_rec-n_t,:] = x_e_new
			x_e_mean_rec[t+n_t_rec-n_t,:] = x_e_mean
			x_ext_rec[t+n_t_rec-n_t,:] = x_ext
			x_ext_mean_rec[t+n_t_rec-n_t,:] = x_ext_mean
			W_rec[t+n_t_rec-n_t,:,:] = W
			W_eext_rec[t+n_t_rec-n_t,:,:] = W_eext
			gain_rec[t+n_t_rec-n_t,:] = gain_neuron
			thresh_rec[t+n_t_rec-n_t,:] = thresh_neuron
			I_ee_rec[t+n_t_rec-n_t,:] = I_ee
			I_eext_rec[t+n_t_rec-n_t,:] = I_eext
		'''
		x_e = x_e_new

	#plt.plot(W_rec[:,0,:])
	#plt.show()

	#plt.plot(x_e_rec)
	#plt.show()
	if store_w and store_w_ext:
		np.savez(path,x_e_rec=x_e_rec,x_e_mean_rec=x_e_mean_rec,x_ext_rec=x_ext_rec,x_ext_mean_rec=x_ext_mean_rec,W_rec=W_rec,W_eext_rec=W_eext_rec,gain_rec=gain_rec,thresh_rec=thresh_rec,I_ee_rec=I_ee_rec,I_eext_rec=I_eext_rec,param_dict=param_dict)
	elif store_w and not(store_w_ext):
		np.savez(path,x_e_rec=x_e_rec,x_e_mean_rec=x_e_mean_rec,x_ext_rec=x_ext_rec,x_ext_mean_rec=x_ext_mean_rec,W_rec=W_rec,W_eext_rec=W_eext,gain_rec=gain_rec,thresh_rec=thresh_rec,I_ee_rec=I_ee_rec,I_eext_rec=I_eext_rec,param_dict=param_dict)
	elif not(store_w) and store_w_ext:
		np.savez(path,x_e_rec=x_e_rec,x_e_mean_rec=x_e_mean_rec,x_ext_rec=x_ext_rec,x_ext_mean_rec=x_ext_mean_rec,W_rec=W,W_eext_rec=W_eext_rec,gain_rec=gain_rec,thresh_rec=thresh_rec,I_ee_rec=I_ee_rec,I_eext_rec=I_eext_rec,param_dict=param_dict)
	else:
		np.savez(path,x_e_rec=x_e_rec,x_e_mean_rec=x_e_mean_rec,x_ext_rec=x_ext_rec,x_ext_mean_rec=x_ext_mean_rec,W_rec=W,W_eext_rec=W_eext,gain_rec=gain_rec,thresh_rec=thresh_rec,I_ee_rec=I_ee_rec,I_eext_rec=I_eext_rec,param_dict=param_dict)
	#pdb.set_trace()
	#with open(path_p,"wb") as file:
	#	pickle.dump(param_dict,file)

	#pdb.set_trace()


if __name__ == "__main__":
	
	file = sys.argv[1]
	
	#file = "no_input"
	#file = "fixed_input_weights"
	#file = "plastic_input_weights"

	param_dict = {}

	with open("./sim_parameters/"+file+".csv","r") as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			param_dict[row[0]] = ast.literal_eval(row[1])

	#pdb.set_trace()
	## Network
	N_e = int(param_dict["N_e"]) # number of excitatory neurons
	sigma_w_init = float(param_dict["sigma_w_init"])
	mu_w_init = float(param_dict["mu_w_init"])

	if param_dict["load_init_data"]:
		init_dat = np.load(param_dict["load_init_data"])
		x_seed = init_dat['x']
		W_seed = init_dat['W']
		gain_seed = init_dat['gain']
		thresh_seed = init_dat['thresh']
	else:

		gain_seed = np.ones(N_e)*float(param_dict["gain_neuron_init"])
		thresh_seed = np.ones(N_e)*float(param_dict["thresh_neuron_init"])

		x_seed = np.random.rand(N_e)
		W_seed = np.random.normal(mu_w_init,sigma_w_init,(N_e,N_e))



	main(x_seed,W_seed,gain_seed,thresh_seed,"./sim_parameters/"+file+".csv","../../misc_data/random_input.ppm","../../misc_data/letter_mapping_simple.csv","./data/"+file)
	
	#x_seed = x_seed+(np.random.rand((N_e))-.5)*0.000000000001
	#W = W+np.random.normal(mu_w_init, sigma_w_init, (N_e,N_e))*0.
	
	#main(x_seed,W_seed,"./data/x_e_rec_2","./data/W_rec_2","./data/I_ee_rec_2","./data/parameters_2.p")
	