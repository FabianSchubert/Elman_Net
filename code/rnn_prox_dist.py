#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, dendrogram

import pdb


###### Parameters

## External Grammar
N_ext = 9 # number of input nodes (features, letters...)
Markov_Gramm = np.zeros((N_ext,N_ext))
Markov_Gramm[3:6,0:3] = 1./3.
Markov_Gramm[6:9,3:6] = 1./3.
Markov_Gramm[0:3,6:9] = 1./3.
##

## Network
N_e = 300 # number of excitatory neurons
N_i = int(N_e*.2) # number of inhibitory neurons

CF_ee = 0.05 # connection fraction E->E
CF_ei = 0.1 # connection fraction I->E
CF_ie = 0.2 # connection fraction E->I
CF_ii = 0.2 # connection fraction I->I

CF_eext = 0.1 # connection fraction Ext->E
w_mean_pre_ext_input = .2

w_exc_min = 0.0001
w_inh_max = -0.0001
##

## Neuron
g_neur = 20. # gain factor of the activation function

r_target_e_mu = 0.1 # mean homeostatic excitatory target firing rate
r_target_e_sigm = 0.#2 # standard deviation of homeostatic excitatory target firing rate
r_target_set_e = np.minimum(1.,np.maximum(0.,np.random.normal(r_target_e_mu,r_target_e_sigm,N_e)))

r_target_i_mu = 0.1 # mean homeostatic inhibitory target firing rate
r_target_i_sigm = 0.#2 # standard deviation of homeostatic inhibitory target firing rate
r_target_set_i = np.minimum(1.,np.maximum(0.,np.random.normal(r_target_i_mu,r_target_i_sigm,N_i))) 

mu_IP = 0.002 # threshold adaption rate

T_e_max_init = 1.
T_i_max_init = 1.

mu_mem_noise = 0.
sigm_mem_noise = np.sqrt(0.01)
##

## Synaptic Normalization
w_total_ee = .5#*N_e**.5 # total presynaptic E->E input
#w_total_eext = .5 # total presynaptic Ext->E input
w_total_ei = -1.#*N_i**.5 # total presynaptic I->E input
w_total_ie = 1.#*N_e**.5 # total presynaptic E->I input
w_total_ii = -1.#*N_i**.5 # total presynaptic I->I input
##

## Excitatory Plasticity
mu_plast_ee = 0.01 # E->E learning rate
##

## Inhibitory Plasticity
mu_plast_ei = 0.01 # I->E learning rate
##

## Simulation
n_t = 20000 # simulation time steps
n_t_skip_w_rec = 100 # only record every n_th weight matrix
n_t_w_rec = int(n_t/n_t_skip_w_rec)
##

## Analysis
t_range_analysis = [n_t-3000,n_t]
t_range_plot_act_raster = [n_t-1000,n_t]
##

######

## Hamming Distance
def hamm_d(x1,x2):
	return np.abs(x1-x2).sum()
##

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

	print ("Initialize I->I")
	while True:
		W_ii = np.random.rand(N_i,N_i)*(np.random.rand(N_i,N_i) <= CF_ii)
		W_ii[np.array(range(N_i)),np.array(range(N_i))] = 0.
		W_ii_conn = 1.*(W_ii != 0.)

		if not 0 in W_ii_conn.sum(axis=1):
			break
	print ("Initialize Ext->I")
	W_eext = 2.*w_mean_pre_ext_input*np.random.rand(N_e,N_ext)*(np.random.rand(N_e,N_ext) <= CF_eext)
	W_eext_conn = 1.*(W_eext != 0.)
	##


	## Initial normalization step
	print("Normalize E->E")
	W_ee = synnorm(W_ee,w_total_ee)
	#print("Normalize Ext->E")
	#W_eext = synnorm(W_eext,w_total_eext)	
	print("Normalize I->E")
	W_ei = synnorm(W_ei,w_total_ei)
	print("Normalize E->I")
	W_ie = synnorm(W_ie,w_total_ie)
	print("Normalize I->I")
	W_ii = synnorm(W_ii,w_total_ii)
	##

	## Initialize activities and temporal storage vectors
	print("Initialize x_e")
	x_e = np.random.rand(N_e)
	print("Initialize x_i")
	x_i = np.random.rand(N_i)

	x_e_old = np.array(x_e)
	x_i_old = np.array(x_i)
	##

	##Initialize Input node
	x_ext = np.zeros((N_ext))
	x_ext[-1] = 1.

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

	ext_sequ_rec = np.ndarray((n_t))
	##

	## Start simulation loop
	for t in tqdm(range(n_t)):

		## Generate Markov Grammar Input
		p_next = np.dot(Markov_Gramm,x_ext)
		next_in = np.random.choice(N_ext,p=p_next)
		x_ext = np.zeros(N_ext)
		x_ext[next_in] = 1.
		##

		## Store old activities (for plasticity mechanisms)
		x_e_old[:] = x_e[:]
		x_i_old[:] = x_i[:]
		##

		## Update activities
		I_eext = np.dot(W_eext,x_ext)

		I_ee = np.dot(W_ee,x_e)
		I_ei = np.dot(W_ei,x_i)

		I_ie = np.dot(W_ie,x_e)
		I_ii = np.dot(W_ii,x_i)

		x_e = s(I_ee + I_ei + I_eext + np.random.normal(mu_mem_noise,sigm_mem_noise,N_e) - T_e,g_neur)
		x_i = s(I_ie + I_ii + np.random.normal(mu_mem_noise,sigm_mem_noise,N_i) - T_i,g_neur)
		##

		## Update tresholds
		T_e = update_th(T_e,x_e,r_target_set_e,mu_IP)
		T_i = update_th(T_i,x_i,r_target_set_i,mu_IP)
		##
		
		
		## Update weights
		W_ee = hebb_ee(W_ee,x_e,x_e_old,mu_plast_ee)
		#W_ei = hebb_ei(W_ei,x_e,x_i_old,r_target_set_e,mu_plast_ei)

		W_ee = synnorm(W_ee,w_total_ee)
		#W_ei = synnorm(W_ei,w_total_ei)

		W_ee = np.maximum(W_ee,w_exc_min)*W_ee_conn
		#W_ei = np.minimum(W_ei,w_inh_max)*W_ei_conn
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

		ext_sequ_rec[t] = next_in
		##

	spt_e = []
	isi_e = []

	isi_e_join = np.ndarray([])

	spt_i = []
	isi_i = []

	isi_i_join = np.ndarray([])

	for k in range(N_e):
		t_sp = np.where(x_e_rec[t_range_analysis[0]:t_range_analysis[1],k] == 1.)[0]
		spt_e.append(t_sp)
		isi_e.append(t_sp[1:]-t_sp[:-1])
		isi_e_join = np.append(isi_e_join,isi_e[-1])

	for k in range(N_i):
		t_sp = np.where(x_i_rec[t_range_analysis[0]:t_range_analysis[1],k] == 1.)[0]
		spt_i.append(t_sp)
		isi_i.append(t_sp[1:]-t_sp[:-1])
		isi_i_join = np.append(isi_i_join,isi_i[-1])

	rec_act_gramm_groups = []



	for k in range(N_ext):
		times_letter = np.where(ext_sequ_rec == k)[0]
		times_letter = times_letter[np.where((times_letter>=t_range_analysis[0])*(times_letter <= t_range_analysis[1]))]
		rec_act_gramm_groups.append(x_e_rec[times_letter,:])

	rec_gramm_groups_mean = np.ndarray((N_ext,N_e))

	for k in range(N_ext):
		rec_gramm_groups_mean[k,:] = rec_act_gramm_groups[k].mean(axis=0)

	Z_m = linkage(rec_gramm_groups_mean,metric=hamm_d)
	#pdb.set_trace()
	Z_full = linkage(x_e_rec[t_range_analysis[0]:t_range_analysis[1],:],method="ward")

	activity_smoothing_kernel = np.exp(-np.array(range(n_t))/100.)
	activity_smoothing_kernel /= activity_smoothing_kernel.sum()

	x_e_smooth = np.convolve(x_e_rec.mean(axis=1),activity_smoothing_kernel)[:n_t]
	x_i_smooth = np.convolve(x_i_rec.mean(axis=1),activity_smoothing_kernel)[:n_t]


	#pdb.set_trace()

	

	#pdb.set_trace()
	
	### Plotting routines	
	fig_sp, ax_sp = plt.subplots(2,1)
	ax_sp[0].pcolormesh(x_e_rec[t_range_plot_act_raster[0]:t_range_plot_act_raster[1],:].T,cmap="gray_r")
	#ax_sp[0].set_xlim([n_t-1000,n_t])
	ax_sp[0].set_xlabel("Time Step + " + str(t_range_plot_act_raster[0]))
	ax_sp[0].set_ylabel("Exc. Neuron #")

	ax_sp[1].pcolormesh(x_i_rec[t_range_plot_act_raster[0]:t_range_plot_act_raster[1],:].T,cmap="gray_r")
	#ax_sp[1].set_xlim([n_t-1000,n_t])
	ax_sp[1].set_xlabel("Time Step + " + str(t_range_plot_act_raster[0]))
	ax_sp[1].set_ylabel("Inh. Neuron #")

	plt.savefig("./plots/act_raster.png",dpi=(300))

	fig_mean_x, ax_mean_x = plt.subplots(1,1)
	ax_mean_x.plot(x_e_smooth,lw=1,label="Excitatory Population")
	ax_mean_x.plot(x_i_smooth,lw=1,label="Inhibitory Population")
	ax_mean_x.legend()
	ax_mean_x.set_xlabel("Time Step")
	ax_mean_x.set_ylabel("Mean Rate (Running Average)")

	plt.savefig("./plots/pop_act_time.png",dpi=(300))

	fig_T,ax_T = plt.subplots(1,1)
	ax_T.plot(T_e_rec.mean(axis=1),label="Excitatory Population")
	ax_T.plot(T_i_rec.mean(axis=1),label="Inhibitory Population")
	ax_T.legend()
	ax_T.set_xlabel("Time Step")
	ax_T.set_ylabel("Mean Treshold")
	
	plt.savefig("./plots/thresholds_time.png",dpi=(300))

	fig_W_ee,ax_W_ee = plt.subplots(1,1)
	ax_W_ee.plot(W_ee_rec[:,0,:],c="k",lw=1)
	ax_W_ee.set_xlabel("Time Step / 10")
	ax_W_ee.set_ylabel("$W^{EE}_{0j}$, $ j \\in \\{ 0,N_e\\}$")

	plt.savefig("./plots/w_ee_sample_time.png",dpi=(300))

	fig_fir_dist, ax_fir_dist = plt.subplots(1,1)
	ax_fir_dist.hist(x_e_rec[t_range_analysis[0]:t_range_analysis[1],:].mean(axis=1),bins=np.linspace(0.,1.,50),histtype="step")
	ax_fir_dist.hist(x_i_rec[t_range_analysis[0]:t_range_analysis[1],:].mean(axis=1),bins=np.linspace(0.,1.,50),histtype="step")
	ax_fir_dist.set_xlabel("Time Average of Firing Rate")
	ax_fir_dist.set_ylabel("Probability")

	plt.savefig("./plots/act_dist.png",dpi=(300))

	fig_isi_dist,ax_isi_dist = plt.subplots(1,1)
	ax_isi_dist.hist(isi_e_join,bins=np.array(range(101)),histtype="step",normed="True",label="Excitatory Population")
	ax_isi_dist.hist(isi_i_join,bins=np.array(range(101)),histtype="step",normed="True",label="Inhibitory Population")
	ax_isi_dist.set_yscale("log")
	ax_isi_dist.set_xlabel("ISI")
	ax_isi_dist.set_ylabel("Probability")
	ax_isi_dist.legend()
	
	plt.savefig("./plots/isi_dist.png",dpi=(300))

	fig_dend = plt.figure()
	dendrogram(Z_m,link_color_func=lambda x:'k')
	plt.xlabel("Input Node Indices")
	plt.ylabel("Hamming Distance")

	plt.savefig("./plots/act_dendrogram_mean.png",dpi=(300))

	fid_dend_full = plt.figure()
	dendrogram(Z_full,p=6,truncate_mode="level",link_color_func=lambda x:'k')
	plt.xlabel("(Truncated) Excitatory Activity Clusters")
	plt.ylabel("Hamming Distance")

	plt.savefig("./plots/act_dendrogram.png",dpi=(300))

	###

	plt.show()
	plt.close("all")
	plt.show()

	## See what we got
	pdb.set_trace()


if __name__ == "__main__":
	main()
