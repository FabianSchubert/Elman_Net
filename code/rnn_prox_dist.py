#!/home/fabian/anaconda3/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


###### Parameters

## Network
N_e = 200 # number of excitatory neurons
N_i = int(N_e*0.2) # number of inhibitory neurons

CF_ee = 0.1 # connection fraction E->E
CF_ei = 0.2 # connection fraction I->E
CF_ie = 0.5 # connection fraction E->I
CF_ii = 0.0 # connection fraction I->I

w_exc_min = 0.001
w_inh_max = -0.001
##

## Neuron
g_neur = 100. # gain factor of the activation function

r_target_e_mu = 0.05 # mean homeostatic excitatory target firing rate
r_target_e_sigm = 0.05 # standard deviation of homeostatic excitatory target firing rate
r_target_set_e = np.minimum(1.,np.maximum(0.,np.random.normal(r_target_e_mu,r_target_e_sigm,N_e)))

r_target_i_mu = 0.1 # mean homeostatic inhibitory target firing rate
r_target_i_sigm = 0.05 # standard deviation of homeostatic inhibitory target firing rate
r_target_set_i = np.minimum(1.,np.maximum(0.,np.random.normal(r_target_i_mu,r_target_i_sigm,N_i))) 

mu_IP = 0.001 # threshold adaption rate

T_e_max_init = 1.
T_i_max_init = 1.

mu_mem_noise = 0.
sigm_mem_noise = 0.#np.sqrt(0.04)
##

## Synaptic Normalization
w_total_ee = .5 # total presynaptic E->E input
w_total_ei = -1. # total presynaptic I->E input
w_total_ie = 1. # total presynaptic E->I input
w_total_ii = 0. # total presynaptic I->I input
##

## Excitatory Plasticity
mu_plast_ee = 0.01 # E->E learning rate
##

## Inhibitory Plasticity
mu_plast_ei = 0.01 # I->E learning rate
##

## Simulation
n_t = 15000 # simulation time steps
n_t_skip_w_rec = 10 # only record every n_th weight matrix
n_t_w_rec = int(n_t/n_t_skip_w_rec)
##

######

## Activation Function
def s(x,gain):
	#return (np.tanh(gain*x/2.)+1.)/2.
	return 1.*(x>0.)
##

## Synaptic Normalization
def synnorm(W,w_total_param):
	sum_in = W.sum(axis=1)
	return (w_total_param*W.T/sum_in).T
##

## EE - Plasticity
def hebb_ee(W,x_new,x_old,mu_learn):
	delta_W = np.outer(x_new,x_old) - np.outer(x_old,x_new)
	delta_W = delta_W*(W!=0.)
	return W + mu_learn*delta_W
##

## EI - Plasticity
def hebb_ei(W,x_e_new,x_i_old,r_target,mu_learn):
	delta_W = np.outer(r_target-x_e_new,x_i_old)
	delta_W = delta_W*(W!=0.)
	return W + mu_learn*delta_W
##

## Threshold Adaptation
def update_th(T,x,r_target,mu):
	return T + mu*(x-r_target)

def main():

	## Initialize Weights
	print("Initialize E->E")
	while True:
		W_ee = np.random.rand(N_e,N_e)*(np.random.rand(N_e,N_e) <= CF_ee)
		W_ee[np.array(range(N_e)),np.array(range(N_e))] = 0.
		W_ee_conn = 1.*(W_ee != 0.)

		if not 0 in W_ee_conn.sum(axis=1):
			break
	print("Initialize I->E")
	while True:
		W_ei = np.random.rand(N_e,N_i)*(np.random.rand(N_e,N_i) <= CF_ei)
		W_ei_conn = 1.*(W_ei != 0.)

		if not 0 in W_ei_conn.sum(axis=1):
			break
	print ("Initialize E->I")
	while True:
		W_ie = np.random.rand(N_i,N_e)*(np.random.rand(N_i,N_e) <= CF_ie)
		W_ie_conn = 1.*(W_ie != 0.)

		if not 0 in W_ie_conn.sum(axis=1):
			break

	W_ii = np.random.rand(N_i,N_i)*(np.random.rand(N_i,N_i) <= CF_ii)
	W_ii[np.array(range(N_i)),np.array(range(N_i))] = 0.
	W_ii_conn = 1.*(W_ii != 0.)
	##


	## Initial normalization step
	print("Normalize E->E")
	W_ee = synnorm(W_ee,w_total_ee)
	print("Normalize I->E")
	W_ei = synnorm(W_ei,w_total_ei)
	print("Normalize E->I")
	W_ie = synnorm(W_ie,w_total_ie)
	#W_ii = synnorm(W_ii,w_total_ii)
	##

	## Initialize activities and temporal storage vectors
	print("Initialize x_e")
	x_e = np.random.rand(N_e)
	print("Initialize x_i")
	x_i = np.random.rand(N_i)

	x_e_old = np.array(x_e)
	x_i_old = np.array(x_i)
	##

	## Initialize thresholds
	T_e = np.random.rand(N_e)*T_e_max_init
	T_i = np.random.rand(N_i)*T_i_max_init

	## Initialize containers for recording
	x_e_rec = np.ndarray((n_t,N_e))
	x_i_rec = np.ndarray((n_t,N_i))

	W_ee_rec = np.ndarray((n_t_w_rec,N_e,N_e))
	W_ei_rec = np.ndarray((n_t_w_rec,N_e,N_i))

	T_e_rec = np.ndarray((n_t,N_e))
	T_i_rec = np.ndarray((n_t,N_i))

	I_ee_rec = np.ndarray((n_t,N_e))
	I_ei_rec = np.ndarray((n_t,N_e))
	##

	## Start simulation loop
	for t in tqdm(range(n_t)):

		## Store old activities (for plasticity mechanisms)
		x_e_old[:] = x_e[:]
		x_i_old[:] = x_i[:]
		##

		## Update activities
		I_ee = np.dot(W_ee,x_e)
		I_ei = np.dot(W_ei,x_i)

		I_ie = np.dot(W_ie,x_e)
		I_ii = np.dot(W_ii,x_i)

		x_e = s(I_ee + I_ei + np.random.normal(mu_mem_noise,sigm_mem_noise,N_e) - T_e,g_neur)
		x_i = s(I_ie + I_ii + np.random.normal(mu_mem_noise,sigm_mem_noise,N_i) - T_i,g_neur)
		##

		## Update tresholds
		T_e = update_th(T_e,x_e,r_target_set_e,mu_IP)
		T_i = update_th(T_i,x_i,r_target_set_i,mu_IP)
		##
		
		## Update weights
		W_ee = hebb_ee(W_ee,x_e,x_e_old,mu_plast_ee)
		W_ei = hebb_ei(W_ei,x_e,x_i_old,r_target_set_e,mu_plast_ei)

		W_ee = synnorm(W_ee,w_total_ee)
		W_ei = synnorm(W_ei,w_total_ei)

		W_ee = np.maximum(W_ee,w_exc_min)*W_ee_conn
		W_ei = np.minimum(W_ei,w_inh_max)*W_ei_conn
		##
	
		## Record
		x_e_rec[t,:] = x_e[:]
		x_i_rec[t,:] = x_i[:]

		if t%n_t_skip_w_rec == 0:
			W_ee_rec[int(t/n_t_skip_w_rec),:,:] = W_ee[:,:]
			W_ei_rec[int(t/n_t_skip_w_rec),:,:] = W_ei[:,:]

		T_e_rec[t,:] = T_e[:]
		T_i_rec[t,:] = T_i[:]

		I_ee_rec[t,:] = I_ee[:]
		I_ei_rec[t,:] = I_ei[:]
		##

	fig_sp, ax_sp = plt.subplots(1,1)
	ax_sp.pcolormesh(x_e_rec[-800:,:].T,cmap="gray_r")
	ax_sp.set_xlabel("Time Step + 4200")
	ax_sp.set_ylabel("Exc. Neuron #")

	fig_mean_x, ax_mean_x = plt.subplots(1,1)
	ax_mean_x.plot(x_e_rec.mean(axis=1),label="Excitatory Population")
	ax_mean_x.plot(x_i_rec.mean(axis=1),label="Inhibitory Population")
	ax_mean_x.legend()
	ax_mean_x.set_xlabel("Time Step")
	ax_mean_x.set_ylabel("Mean Rate")

	fig_T,ax_T = plt.subplots(1,1)
	ax_T.plot(T_e_rec.mean(axis=1),label="Excitatory Population")
	ax_T.plot(T_i_rec.mean(axis=1),label="Inhibitory Population")
	ax_T.legend()
	ax_T.set_xlabel("Time Step")
	ax_T.set_ylabel("Mean Treshold")
	

	plt.show()


	## See what we got
	pdb.set_trace()


if __name__ == "__main__":
	main()