#!/usr/bin/env python3

from data_analysis import *

import pdb
from tqdm import tqdm

import sys
import os

def angle(v1,v2):
	return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


file = sys.argv[1]
#file = "no_input"
#file = "fixed_input_weights"
#file = "plastic_input_weights"

if not os.path.isdir("./plots/"+file):
	os.makedirs("./plots/"+file)


x,W_ee,W_eext,I_ee,I_eext,p = load_data("./data/"+file+".npz")
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
fig,ax = plt.subplots(1,1,figsize=(15,5))

ax.plot(t_ax,x)
ax.set_xlabel("#t")
ax.set_ylabel("$x_i$")

ax.set_xlim([t_ax[0],t_ax[-1]])

plt.tight_layout()

plt.savefig("./plots/"+file+"/activity_plot.png",dpi=300)

plt.show()
'''
l = np.ndarray((p["n_t_rec"],p["N_e"]),dtype="complex128")



for k in tqdm(range(p["n_t_rec"])):
	d_sigm = d_act(np.dot(W_ee[k,:,:],x[k,:]),p["gain_neuron"])
	W_tilde = (W_ee[k,:,:].T*d_sigm).T
	l[k,:] = np.linalg.eig(W_tilde)[0]

max_l_real_abs = np.abs(np.real(l)).max(axis=1)
plt.plot(t_ax,np.log(max_l_real_abs),'-o')
plt.grid()
plt.xlabel("#t")
plt.ylabel("Max. Lyapunov Exponent")
plt.show()

#pdb.set_trace()
'''
fig,ax = plt.subplots(1,1,figsize=(15,5))

ax.plot(t_ax,W_ee[:,0,:],c='k')
ax.plot(t_ax,W_eext[:,0,:],c='b')

ax.set_xlabel("#t")
ax.set_ylabel("$W_{0j}$ (black),$W_{0j,\\rm ext}$ (blue)")

ax.set_xlim([t_ax[0],t_ax[-1]])

plt.tight_layout()

plt.savefig("./plots/"+file+"/weights_repr_plot.png",dpi=300)

plt.show()

n_bins = 100
bins_h = np.linspace(-5.,5.,n_bins+1)
h = np.ndarray((p["n_t_rec"],n_bins))
h_ext = np.ndarray((p["n_t_rec"],n_bins))

for k in range(p["n_t_rec"]):
	h_temp = np.histogram(np.reshape(W_ee[k,:,:],(p["N_e"]**2)),bins=bins_h)
	h_temp_ext = np.histogram(np.reshape(W_eext[k,:,:],(p["N_e"]*p["N_ext"])),bins=bins_h)
	h[k,:] = h_temp[0]
	h_ext[k,:] = h_temp_ext[0]

fig,ax = plt.subplots(1,2,figsize=(15,5))
ax[0].pcolormesh(t_ax,bins_h,h.T)
ax[1].pcolormesh(t_ax,bins_h,h_ext.T)

ax[0].set_ylabel("$W_{ij}$")
ax[1].set_ylabel("$W_{ij,\\rm ext}$")

ax[0].set_xlabel("#t")
ax[1].set_xlabel("#t")

plt.tight_layout()

plt.savefig("./plots/"+file+"/weights_hist_time.png",dpi=300)

plt.show()



I_ee_mean = I_ee.mean(axis=1)
I_ee_std = I_ee.std(axis=1)

I_eext_mean = I_eext.mean(axis=1)
I_eext_std = I_eext.std(axis=1)

fig, ax = plt.subplots(1,1,figsize=(15,5))
ax.fill_between(t_ax,I_ee_mean-I_ee_std,I_ee_mean+I_ee_std,color='k',lw=0,alpha=0.5)
ax.fill_between(t_ax,I_eext_mean-I_eext_std,I_eext_mean+I_eext_std,color='b',lw=0,alpha=0.5)

ax.plot(t_ax,I_ee_mean,color='k')
ax.plot(t_ax,I_eext_mean,color='b')

ax.set_xlim([t_ax[0],t_ax[-1]])

ax.set_xlabel("#t")
ax.set_ylabel("$W_{ij}$ (black),$W_{ij,\\rm ext}$ (blue)")

plt.tight_layout()

plt.savefig("./plots/"+file+"/I_ee_I_eext_mean_std.png",dpi=300)

plt.show()
#plt.plot(W_ee.std(axis=2))

#plt.show()

pdb.set_trace()

#animate_mem_dist(I1,p)
