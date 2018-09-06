#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from plot_setting import *

import pdb

def sigm(x):

	return (np.tanh(x/2.)+1.)/2.


def act_pd(p,d,alpha,gain):

	return sigm(gain*d) + alpha*sigm(gain*p)*sigm(-gain*d)


def gen_rand_sequ(N,T,dt,n_out,g=2.):

	n_t = int(1.2*T/dt)

	W = np.random.normal(0.,g/N**.5,(N,N))

	x = np.random.normal(0.,.5,(N))

	x_rec = np.ndarray((n_t,n_out))

	for t in tqdm(range(n_t)):

		x += dt*(-x + np.tanh(np.dot(W,x)))

		x_rec[t,:] = x[:n_out]

	return x_rec[int(0.2/1.2*n_t):,:]

def synnorm(w,w_total):
	#return w_total * w/w.sum()
	return w_total * w/np.linalg.norm(w)



def main(n_prox = 1, n_dist = 10, n_t_learn = 2000000, X_p = np.load("rand_chaotic_sequ.npy"), X_d = np.load("rand_chaotic_sequ.npy")[:,0], alpha_pd = 0.2, gain_pd = 20., gain_d_sign_inv = 1., w_dist = 1., w_prox_total = 1., w_prox_max = 1., w_prox_min = 0.0001, w_dist_total = 1., w_dist_max = 1., w_dist_min = 0.0001, mu_learn = 0.00005, mu_hom = 0.00002,  mu_avg = 0.00002):
	
	# initialize proximal weights
	w_prox = np.ones(n_prox)
	w_prox = w_prox_total * w_prox/w_prox.sum()

	# initialize distal weights
	w_dist = np.ones(n_dist)
	w_dist = w_dist_total * w_dist/w_dist.sum()

	# initialize weights for "analytic" time evolution, by covariances
	w_prox_analytic = np.ones(n_prox)
	w_prox_analytic = w_prox_total * w_prox_analytic/w_prox_analytic.sum()

	gamma = (1.-alpha_pd/2.)*alpha_pd*gain_pd/4.

	'''
	# Calc covariance matrices
	C_xx = np.cov(X_p.T)
	C_xd = np.cov(X_p.T,X_d)[-1,:-1]

	C_xxd = np.ndarray((n_prox,n_prox))

	for i in range(n_prox):
		for j in range(n_prox):
			
			C_xxd[i,j] = ((X_d - X_d.mean())*(X_p[:,i] - X_p[:,i].mean())*(X_p[:,j] - X_p[:,j].mean())).mean()

	'''

	th_p = 0.
	th_d = 0.

	# initialize running averages
	x_mean = 0.5
	X_p_mean = X_p[0,:]
	X_d_mean = X_d[0,:]

	# initialize recordings
	x_rec = np.ndarray((n_t_learn))
	x_mean_rec = np.ndarray((n_t_learn))

	w_prox_rec = np.ndarray((n_t_learn,n_prox))
	w_prox_analytic_rec = np.ndarray((n_t_learn,n_prox))

	'''
	w_prox_analytic_cxx_rec = np.ndarray((n_t_learn,n_prox))
	w_prox_analytic_cxd_rec = np.ndarray((n_t_learn,n_prox))
	w_prox_analytic_cxxd_rec = np.ndarray((n_t_learn,n_prox))
	'''

	w_dist_rec = np.ndarray((n_t_learn,n_dist))

	X_p_mean_rec = np.ndarray((n_t_learn,n_prox))
	X_d_mean_rec = np.ndarray((n_t_learn,n_dist))	

	I_p_rec = np.ndarray((n_t_learn))
	I_d_rec = np.ndarray((n_t_learn))

	th_p_rec = np.ndarray((n_t_learn))
	th_d_rec = np.ndarray((n_t_learn))

	for t in tqdm(range(n_t_learn)):

		I_p = np.dot(w_prox,X_p[t,:]) - th_p
		I_d = np.dot(w_dist,X_d[t,:]) - th_d

		th_p += mu_hom*I_p
		th_d += mu_hom*I_d
		
		x = act_pd(I_p,I_d,alpha_pd,gain_pd)

		x_mean += mu_avg*(x - x_mean)
		X_p_mean += mu_avg*(X_p[t,:] - X_p_mean)
		X_d_mean += mu_avg*(X_d[t,:] - X_d_mean)

		## plasticity
		
		#w_prox += mu_learn * (x-x_mean)*(X_p[t,:]-X_p_mean)

		#w_prox = np.maximum(w_prox_min,w_prox)
		#w_prox = np.minimum(w_prox_max,w_prox)
		#w_prox = w_prox_total * w_prox/w_prox.sum()
		#w_prox = synnorm(w_prox,w_prox_total)

		w_dist += mu_learn * (x-x_mean)*(X_d[t,:]-X_d_mean)

		#w_dist = np.maximum(w_dist_min,w_dist)
		#w_dist = np.minimum(w_dist_max,w_dist)
		#w_dist = w_dist_total * w_dist/w_dist.sum()
		w_dist = synnorm(w_dist,w_dist_total)
		##


		'''
		## plasticity-analytic
		w_prox_analytic += mu_learn * (alpha_pd*np.dot(C_xx,w_prox_analytic) + (2.-alpha_pd)*C_xd + gamma*np.dot(C_xxd,w_prox_analytic)) * gain_pd/8.

		#w_prox_analytic += mu_learn * (alpha_pd*np.dot(C_xx,w_prox_analytic) + (2.-alpha_pd)*C_xd) * gain_pd/8.

		w_prox_analytic_cxx_rec[t,:] = alpha_pd*np.dot(C_xx,w_prox_analytic)
		w_prox_analytic_cxd_rec[t,:] = (2.-alpha_pd)*C_xd
		w_prox_analytic_cxd_rec[t,:] = gamma*np.dot(C_xxd,w_prox_analytic)

		w_prox_analytic = np.maximum(w_prox_min,w_prox_analytic)
		w_prox_analytic = np.minimum(w_prox_max,w_prox_analytic)
		w_prox_analytic = w_prox_total * w_prox_analytic/w_prox_analytic.sum()
		##
		'''

		x_rec[t] = x
		x_mean_rec[t] = x_mean

		w_prox_rec[t,:] = w_prox
		w_prox_analytic_rec[t,:] = w_prox_analytic

		w_dist_rec[t,:] = w_dist

		X_p_mean_rec[t,:] = X_p_mean

		I_p_rec[t] = I_p
		I_d_rec[t] = I_d

		th_p_rec[t] = th_p
		th_d_rec[t] = th_d

	if __name__ == "__main__":
		
		print("plotting...")

		t_ax = np.array(range(n_t_learn))


		###############
		fig_act_pd, ax_act_pd = plt.subplots(1,2,figsize=(default_fig_width,2.5))
		i_p = np.linspace(-1.,1.,400)
		i_d = np.linspace(-1.,1.,400)
		Ip,Id = np.meshgrid(i_p,i_d)

		act_pd_p_beginning = ax_act_pd[0].pcolormesh(i_p,i_d,act_pd(Ip,Id,alpha_pd,gain_pd))

		ax_act_pd[0].set_xlim([-1.,1.])
		ax_act_pd[0].set_ylim([-1.,1.])

		plt.colorbar(mappable=act_pd_p_beginning)

		t_wind = int(n_t_learn*0.02)
		ax_act_pd[0].plot(I_p_rec[:t_wind],I_d_rec[:t_wind],'.',c='r',alpha=0.2)
		ax_act_pd[0].set_xlabel("$I_{prox}$")
		ax_act_pd[0].set_ylabel("$I_{dist}$")

		act_pd_p_end = ax_act_pd[1].pcolormesh(i_p,i_d,act_pd(Ip,Id,alpha_pd,gain_pd))

		ax_act_pd[1].set_xlim([-1.,1.])
		ax_act_pd[1].set_ylim([-1.,1.])

		ax_act_pd[1].plot(I_p_rec[-t_wind:],I_d_rec[-t_wind:],'.',c='r',alpha=0.2)
		ax_act_pd[1].set_xlabel("$I_{prox}$")
		ax_act_pd[1].set_ylabel("$I_{dist}$")

		ax_act_pd[0].set_title("First " + str(int(100.*t_wind/n_t_learn))+"% of learning phase",fontsize=10)
		ax_act_pd[1].set_title("Last " + str(int(100.*t_wind/n_t_learn))+"% of learning phase",fontsize=10)

		plt.tight_layout()
		###############

		###############
		fig_w_prox, ax_w_prox = plt.subplots(figsize=(default_fig_width/2.,default_fig_width/3.))
		ax_w_prox.plot(w_prox_rec)
		ax_w_prox.set_xlabel("#t")
		ax_w_prox.set_ylabel("$w_{prox}$")

		ax_w_prox.get_xaxis().set_ticks(np.linspace(0,n_t_learn,3))

		ax_w_prox.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)

		plt.tight_layout()
		###############

		###############
		fig_w_dist, ax_w_dist = plt.subplots(figsize=(default_fig_width/2.,default_fig_width/3.))
		ax_w_dist.plot(w_dist_rec)
		ax_w_dist.set_xlabel("#t")
		ax_w_dist.set_ylabel("$w_{dist}$")

		ax_w_dist.get_xaxis().set_ticks(np.linspace(0,n_t_learn,3))

		ax_w_dist.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)

		plt.tight_layout()
		###############

		'''*
		###############
		fig_w_analytic, ax_w_analytic = plt.subplots(figsize=(5.,2.5))
		ax_w_analytic.plot(w_prox_analytic_rec)
		ax_w_analytic.set_xlabel("#t")
		ax_w_analytic.set_ylabel("$w_{prox analytic}$")

		plt.tight_layout()
		###############
		'''

		###############
		fig_I, ax_I = plt.subplots(2,1,figsize=(default_fig_width,4))
		t_wind = int(n_t_learn*0.01)

		ax_I[0].plot(I_d_rec[:t_wind],label="$I_d$")
		ax_I[0].plot(I_p_rec[:t_wind],label="$I_p$")

		ax_I[0].legend()

		ax_I[0].get_xaxis().set_ticks(np.linspace(0,t_wind,3))
		ax_I[0].ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)	

		ax_I[0].set_xlabel("#t")
		ax_I[0].set_ylabel("$I_{prox}$, $I_{dist}$")
		ax_I[0].set_title("First " + str(int(100.*t_wind/n_t_learn))+"% of learning phase",fontsize=10)

		ax_I[1].plot(t_ax[-t_wind:],I_d_rec[-t_wind:])
		ax_I[1].plot(t_ax[-t_wind:],I_p_rec[-t_wind:])

		ax_I[1].get_xaxis().set_ticks(np.linspace(n_t_learn-t_wind,n_t_learn,3))
		ax_I[1].ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)

		ax_I[1].set_xlabel("#t")
		ax_I[1].set_ylabel("$I_{prox}$, $I_{dist}$")
		ax_I[1].set_title("Last " + str(int(100.*t_wind/n_t_learn))+"% of learning phase",fontsize=10)


		plt.tight_layout()
		###############

		###############
		t_wind = int(n_t_learn*0.005)
		fig_X_p, ax_X_p = plt.subplots(figsize=(default_fig_width,2.))
		ax_X_p.plot(X_p[:t_wind])
		ax_X_p.set_xlabel("#t")
		ax_X_p.set_ylabel("$x_{prox}$")

		ax_X_p.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)

		plt.tight_layout()
		###############

		###############
		t_wind = int(n_t_learn*0.005)
		fig_X_d, ax_X_d = plt.subplots(figsize=(default_fig_width,2.))
		ax_X_d.plot(X_d[:t_wind])
		ax_X_d.set_xlabel("#t")
		ax_X_d.set_ylabel("$x_{dist}$")

		ax_X_d.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)

		plt.tight_layout()
		###############


		fold = "sequ_learn/"

		fig_act_pd.savefig(plots_base_folder + fold + "act_pd.png",dpi=300)
		fig_w_prox.savefig(plots_base_folder + fold + "w_prox.png",dpi=300)
		fig_w_dist.savefig(plots_base_folder + fold + "w_dist.png",dpi=300)
		fig_I.savefig(plots_base_folder + fold + "I.png",dpi=300)
		fig_X_p.savefig(plots_base_folder + fold + "X_p.png",dpi=300)
		fig_X_d.savefig(plots_base_folder + fold + "X_d.png",dpi=300)

		plt.show()

		pdb.set_trace()


if __name__ == "__main__":
	
	#'''
	X_p = np.ndarray((2000000,10))
	for k in tqdm(range(10)):

		X_p[:,k] = gen_rand_sequ(500,2000000*0.1,0.1,1,2.)[:,0]

	X_p = (X_p+1.)/2.

	np.save("rand_chaotic_sequ.npy",X_p)
	#'''	

	X_d_sequ = np.load("rand_chaotic_sequ.npy")

	#X_d_sequ = np.ndarray((2000000,10))

	X_p_sequ = X_p_sequ[:,0]

		
	main(X_p = X_p_sequ, X_d = X_d_sequ)







	



