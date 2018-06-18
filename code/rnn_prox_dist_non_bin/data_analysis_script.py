#!/usr/bin/env python3

from data_analysis import *
from matplotlib import gridspec

import pdb
from tqdm import tqdm

import sys
import os


def gauss(x,mean,std):

	return np.exp(-(x-mean)**2/(2.*std**2))/(np.sqrt(2.*np.pi)*std)

def angle(v1,v2):
	return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


file = sys.argv[1]
#file = "no_input"
#file = "fixed_input_weights"
#file = "plastic_input_weights"

if not os.path.isdir("./plots/"+file):
	os.makedirs("./plots/"+file)


dat = load_data("./data/"+file+".npz")

if "W" in dat:
	W_ee = dat["W"]
if "W_eext" in dat:
	W_eext = dat["W_eext"]


x = dat["x"]
x_mean = dat["x_mean"]
x_ext = dat["x_ext"]
x_ext_mean = dat["x_ext_mean"]
gain = dat["gain"]
thresh = dat["thresh"]
I_ee = dat["I_ee"]
I_eext = dat["I_eext"]
p = dat["p"]
#x2,W2,I2,p2 = load_data("./data/x_e_rec_2.npy","./data/W_rec_2.npy","./data/I_ee_rec_2.npy","./data/parameters_2.p")



t_ax = np.linspace(0.,p["n_t"],p["n_t_rec"])

plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


#dist = (((x - x2)**2).sum(axis=1))**.5

#plt.plot(dist,'-o')
#plt.yscale("log")
#plt.show()

'''

angles_end = []
angles_start = []

for k in range(p["N_ext"]):
	for l in range(k):
		angles_start.append(angle(W_eext[0,:,k],W_eext[0,:,l])*360./(2.*np.pi))
		angles_end.append(angle(W_eext[-1,:,k],W_eext[-1,:,l])*360./(2.*np.pi))

angles_start = np.array(angles_start)
angles_end = np.array(angles_end)

plt.hist(angles_start,bins=30,normed=True,histtype="step")
plt.hist(angles_end,bins=30,normed=True,histtype="step")

plt.show()
'''

def plot_act(skip_dat = 1):
	fig_act,ax_act = plt.subplots(1,1,figsize=(15,5))

	ax_act.plot(t_ax[::skip_dat],x[::skip_dat,:])
	ax_act.set_xlabel("#t")
	ax_act.set_ylabel("$x_i$")

	ax_act.set_xlim([t_ax[0],t_ax[-1]])

	plt.tight_layout()

	plt.savefig("./plots/"+file+"/activity_plot.png",dpi=300)

	plt.show()



def plot_lyap_exp():

	l = np.ndarray((p["n_t_rec"],p["N_e"]),dtype="complex128")



	for k in tqdm(range(p["n_t_rec"])):
		d_sigm = x[k,:]*(1.-x[k,:])*gain[k,:]#d_act(np.dot(W_ee[k,:,:],x[k,:]),gain[k,:])
		W_tilde = (W_ee[k,:,:].T*d_sigm).T
		l[k,:] = np.linalg.eig(W_tilde)[0]

	max_l_real_abs = np.abs(np.real(l)).max(axis=1)

	fig_lyap_exp,ax_lyap_exp = plt.subplots(1,1,figsize=(15,5))

	ax_lyap_exp.plot(t_ax,np.log(max_l_real_abs),'-o')
	ax_lyap_exp.grid()
	ax_lyap_exp.set_xlabel("#t")
	ax_lyap_exp.set_ylabel("Max. Lyapunov Exponent")
	
	plt.savefig("./plots/" + file + "/lyapunov_exponents.png",dpi=300)

	plt.show()



def plot_weights_sample():

	fig_weights_sample,ax_weights_sample = plt.subplots(1,1,figsize=(15,5))

	ax_weights_sample.plot(t_ax,W_ee[:,0,:],c='k')
	ax_weights_sample.plot(t_ax,W_eext[:,0,:],c='b')

	ax_weights_sample.set_xlabel("#t")
	ax_weights_sample.set_ylabel("$W_{0j}$ (black),$W_{0j,\\rm ext}$ (blue)")

	ax_weights_sample.set_xlim([t_ax[0],t_ax[-1]])

	plt.tight_layout()

	plt.savefig("./plots/"+file+"/weights_repr_plot.png",dpi=300)

	plt.show()


def plot_weights_hist(n_bins=100):
	
	bins_h = np.linspace(-5.,5.,n_bins+1)
	h = np.ndarray((p["n_t_rec"],n_bins))
	h_ext = np.ndarray((p["n_t_rec"],n_bins))

	for k in range(p["n_t_rec"]):
		h_temp = np.histogram(np.reshape(W_ee[k,:,:],(p["N_e"]**2)),bins=bins_h)
		h_temp_ext = np.histogram(np.reshape(W_eext[k,:,:],(p["N_e"]*p["N_ext"])),bins=bins_h)
		h[k,:] = h_temp[0]
		h_ext[k,:] = h_temp_ext[0]

	fig_weights_hist,ax_weights_hist = plt.subplots(1,2,figsize=(15,5))
	ax_weights_hist[0].pcolormesh(t_ax,bins_h,h.T)
	ax_weights_hist[1].pcolormesh(t_ax,bins_h,h_ext.T)

	ax_weights_hist[0].set_ylabel("$W_{ij}$")
	ax_weights_hist[1].set_ylabel("$W_{ij,\\rm ext}$")

	ax_weights_hist[0].set_xlabel("#t")
	ax_weights_hist[1].set_xlabel("#t")

	plt.tight_layout()

	plt.savefig("./plots/"+file+"/weights_hist_time.png",dpi=300)

	plt.show()


def plot_I_mean_std():
	I_ee_mean = I_ee.mean(axis=1)
	I_ee_std = I_ee.std(axis=1)

	I_eext_mean = I_eext.mean(axis=1)
	I_eext_std = I_eext.std(axis=1)

	fig_I_mean_std, ax_I_mean_std = plt.subplots(1,1,figsize=(15,5))
	ax_I_mean_std.fill_between(t_ax,I_ee_mean-I_ee_std,I_ee_mean+I_ee_std,color='k',lw=0,alpha=0.5)
	ax_I_mean_std.fill_between(t_ax,I_eext_mean-I_eext_std,I_eext_mean+I_eext_std,color='b',lw=0,alpha=0.5)

	ax_I_mean_std.plot(t_ax,I_ee_mean,color='k')
	ax_I_mean_std.plot(t_ax,I_eext_mean,color='b')

	ax_I_mean_std.set_xlim([t_ax[0],t_ax[-1]])

	ax_I_mean_std.set_xlabel("#t")
	ax_I_mean_std.set_ylabel("$W_{ij}$ (black),$W_{ij,\\rm ext}$ (blue)")

	plt.tight_layout()

	plt.savefig("./plots/"+file+"/I_ee_I_eext_mean_std.png",dpi=300)

	plt.show()


def plot_I_ee_I_eext_diff():
	fig_I_ee_I_eext_diff, ax_I_ee_I_eext_diff = plt.subplots(1,1,figsize=(15,5))

	ax_I_ee_I_eext_diff.plot(t_ax,(np.abs(I_ee)-np.abs(I_eext))/np.abs(I_ee+I_eext))

	ax_I_ee_I_eext_diff.set_xlabel("#t")
	ax_I_ee_I_eext_diff.set_ylabel("$(|I_{ee}|-|I_{eext}|)/|I_{ee}+I_{eext}|$")

	plt.tight_layout()

	plt.savefig("./plots/"+file+"/I_ee_I_eext_difference.png",dpi=300)

	plt.show()


def plot_I_hist(n_bins=50):


	bins_h = np.linspace(-15.,15.,n_bins+1)
	h = np.ndarray((p["n_t_rec"],n_bins))
	h_ext = np.ndarray((p["n_t_rec"],n_bins))

	for k in range(p["n_t_rec"]):
		h_temp = np.histogram(I_ee[k,:],bins=bins_h)
		h_temp_ext = np.histogram(I_eext[k,:],bins=bins_h)
		h[k,:] = h_temp[0]
		h_ext[k,:] = h_temp_ext[0]

	fig_I_hist = plt.figure(figsize=(15,5))

	gs = gridspec.GridSpec(1,4, width_ratios=[3,1,3,1])

	ax_I_hist = [plt.subplot(gs[k]) for k in range(4)]

	ax_I_hist[0].pcolormesh(t_ax,bins_h,h.T)
	ax_I_hist[2].pcolormesh(t_ax,bins_h,h_ext.T)
	#pdb.set_trace()
	ax_I_hist[1].step(h[-1,:],bins_h[1:])
	ax_I_hist[3].step(h_ext[-1,:],bins_h[1:])

	ax_I_hist[0].set_ylabel("$I_{ee}$")
	ax_I_hist[2].set_ylabel("$I_{eext}$")

	ax_I_hist[0].set_xlabel("#t")
	ax_I_hist[2].set_xlabel("#t")

	ax_I_hist[1].set_xlabel("p")
	ax_I_hist[3].set_xlabel("p")


	plt.tight_layout()

	plt.savefig("./plots/"+file+"/input_hist_time.png",dpi=300)

	plt.show()

def analyze_fp_stab():

	d_fix = p["gain_neuron"]*p["act_fix"]*(1.-p["act_fix"])

	l = np.ndarray((p["n_t_rec"]))

	for k in tqdm(range(p["n_t_rec"])):

		l[k] = np.log(np.abs(np.real(np.linalg.eigvals(d_fix*W_ee[k,:,:]))).max())

	plt.plot(t_ax,l,'-o')

	plt.savefig("./plots/" + file + "/fp_stability.png",dpi=300)

	plt.show()


def plot_rec_field_end():

	fig_rec_field_end, ax_rec_field_end = plt.subplots(1,1,figsize=(8,8))

	ax_rec_field_end.pcolormesh(np.abs(W_eext[-1,:,:]))
	
	
	ax_rec_field_end.set_ylabel("$x_{\\mathrm{e},i}$")
	ax_rec_field_end.set_xlabel("$x_{\\mathrm{ext},i}$")

	ax_rec_field_end.xaxis.tick_top()
	ax_rec_field_end.xaxis.set_label_position("top")
	ax_rec_field_end.invert_yaxis()

	plt.show()

def comp_act_target_dist(n_last_steps):

	fig_comp_act_target_dist, ax_comp_act_target_dist = plt.subplots(1,1,figsize=(5,5))

	ax_comp_act_target_dist.hist(np.reshape(x[-n_last_steps:,:],(n_last_steps*p["N_e"])),bins=50,normed=True)

	x_space = np.linspace(0.,1.,1000)

	ax_comp_act_target_dist.plot(x_space,gauss(x_space,p["mean_act_target"],p["std_act_target"]))

	ax_comp_act_target_dist.set_ylabel("$p$")
	ax_comp_act_target_dist.set_xlabel("$x_e$")

	plt.show()




#plot_I_hist()
#analyze_fp_stab()
plt.ion()
pdb.set_trace()

#animate_mem_dist(I1,p)
