#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import pdb

def act(x):

	return np.tanh(x)

def act_pos(x):

	return (np.tanh(2.*x)+1.)/2.

def act_pd(p,d,gain,alpha):

	return (alpha + (1.-alpha)*act_pos(d*gain))*act_pos(p*gain)

def act_pd_2(p,d,gain,theta_p0,theta_p1,theta_d,alpha):

	return act_pos(gain*(p-theta_p1))*act_pos(gain*(d-theta_d)) + alpha*act_pos(gain*(p-theta_p0))*act_pos(-gain*(d-theta_d))

def gen_rand_sequ(N,T,dt,n_out,g=2.):

	n_t = int(1.2*T/dt)

	W = np.random.normal(0.,g/N**.5,(N,N))

	x = np.random.normal(0.,.5,(N))

	x_rec = np.ndarray((n_t,n_out))

	for t in tqdm(range(n_t)):

		x += dt*(-x + act(np.dot(W,x)))

		x_rec[t,:] = x[:n_out]

	return x_rec[int(0.2/1.2*n_t):,:]


def gen_rand_sequ_ou(N,T,dt,sigma,theta):

	n_t = int(1.2*T/dt)

	x = np.random.normal(0.,.5,(N))

	x_rec = np.ndarray((n_t,N))

	for t in tqdm(range(n_t)):

		x += dt*(-x*theta) + dt**.5 * np.random.normal(0.,1.,(N))*sigma
		x_rec[t,:] = x[:]

	return x_rec[int(0.2/1.2*n_t):,:]


n = 10

n_t_learn = 500000

w_prox_total = 1.
w_prox_max = w_prox_total
w_prox_min = 0.

w_prox = np.ones(n)
w_prox[0] = 0.1

w_prox = w_prox_total * w_prox/w_prox.sum()

w_dist = 1.


X_p = np.load("rand_chaotic_sequ.npy")

'''
X_p = np.ndarray((n_t_learn,n))
for k in tqdm(range(n)):

	X_p[:,k] = gen_rand_sequ(500,n_t_learn*0.1,0.1,1,2.)[:,0]

X_p = (X_p+1.)/2.
'''

#X_p = np.random.rand(n_t_learn,n)

#X_p = gen_rand_sequ_ou(n,n_t_learn*0.1,0.1,0.2,0.2)
#X_p = 0.25*(X_p - X_p.mean())/X_p.std() + 0.5

#X_p = 1.*(np.random.rand(n_t_learn,n) <= 0.1)

#X_d = 1.*(np.random.rand(n_t_learn) <= 0.5*(X_p[:,0]+X_p[:,1]))

'''
lin_comb = np.random.rand(n)
lin_comb /= lin_comb.sum()
X_d = np.dot(X_p,lin_comb)
'''


X_p[:,1] *= 3.
#X_p[:,1:5] *=0.5
X_d = X_p[:,0]


mu_learn = 0.0001

alpha_pd = 0.05

gain_pd = 5.

x_mean = 0.5
mu_x_mean = 0.0001

X_p_mean = 0.5*np.ones(n)
mu_X_p_mean = 0.0001

I_p_mean = 0.5
I_d_mean = 0.5
mu_I_p_mean = 0.0001
mu_I_d_mean = 0.0001

x_rec = np.ndarray((n_t_learn))
x_mean_rec = np.ndarray((n_t_learn))

w_prox_rec = np.ndarray((n_t_learn,n))

I_p_rec = np.ndarray((n_t_learn))
I_d_rec = np.ndarray((n_t_learn))

I_p_mean_rec = np.ndarray((n_t_learn))
I_d_mean_rec = np.ndarray((n_t_learn))



for t in tqdm(range(n_t_learn)):

	I_p = np.dot(w_prox,X_p[t,:])
	I_d = w_dist * X_d[t]

	th_p0 = I_p_mean#0.7*I_p_mean
	th_p1 = 0.1*I_p_mean
	th_d = 1.4*I_d_mean

	#x = act_pd(I_p-th_p,I_d-th_d,gain_pd,alpha_pd)
	#x = act_pd_2(I_p,I_d,gain_pd,th_p0,th_p1,th_d,alpha_pd)
	x = act_pos(I_p+I_d-0.5)
	#x = (I_p+I_d)
	#x = act_pos(I_p*I_d)
	#x = (I_p*I_d)
	#x = (I_p+I_d)/(1.+20.*(I_p-I_d)**2)


	x_mean += mu_x_mean*(-x_mean + x)

	I_p_mean += mu_I_p_mean*(-I_p_mean + I_p)
	I_d_mean += mu_I_d_mean*(-I_d_mean + I_d)

	X_p_mean += mu_X_p_mean*(-X_p_mean + X_p[t,:])

	#w_prox += mu_learn*(X_p[t,:]*x*(x - fp_dep))
	w_prox += mu_learn * (x-x_mean)*(X_p[t,:]-X_p_mean)
	#w_prox = w_prox + (1. - w_prox.sum())/n

	w_prox = np.maximum(w_prox_min,w_prox)
	w_prox = np.minimum(w_prox_max,w_prox)
	w_prox = w_prox_total * w_prox/w_prox.sum()
	#

	x_rec[t] = x
	x_mean_rec[t] = x_mean
	w_prox_rec[t,:] = w_prox[:]

	I_p_rec[t] =I_p
	I_d_rec[t] = I_d

	I_p_mean_rec[t] =I_p_mean
	I_d_mean_rec[t] = I_d_mean


t_ax = np.array(range(n_t_learn))

###
fig_act_pd, ax_act_pd = plt.subplots(1,2,figsize=(10.,4.))
i_p = np.linspace(0.,2.5,400)
i_d = np.linspace(0.,2.5,400)
Ip,Id = np.meshgrid(i_p,i_d)

#act_pd_p_beginning = ax_act_pd[0].pcolormesh(i_p,i_d,act_pd(Ip-th_p,Id-th_d,gain_pd,alpha_pd))

#act_pd_p_beginning = ax_act_pd[0].pcolormesh(i_p,i_d,act_pd_2(Ip,Id,gain_pd,th_p0,th_p1,th_d,alpha_pd))
act_pd_p_beginning = ax_act_pd[0].pcolormesh(i_p,i_d,act_pos(Ip+Id-.5))

plt.colorbar(mappable=act_pd_p_beginning)

t_wind = int(n_t_learn*0.02)
ax_act_pd[0].plot(I_p_rec[:t_wind],I_d_rec[:t_wind],'.',c='r',alpha=0.2)
ax_act_pd[0].set_xlabel("$I_{prox}$")
ax_act_pd[0].set_ylabel("$I_{dist}$")

#act_pd_p_end = ax_act_pd[1].pcolormesh(i_p,i_d,act_pd(Ip-th_p,Id-th_d,gain_pd,alpha_pd))

#act_pd_p_end = ax_act_pd[1].pcolormesh(i_p,i_d,act_pd_2(Ip,Id,gain_pd,th_p0,th_p1,th_d,alpha_pd))
act_pd_p_end = ax_act_pd[1].pcolormesh(i_p,i_d,act_pos(Ip+Id-.5))


ax_act_pd[1].plot(I_p_rec[-t_wind:],I_d_rec[-t_wind:],'.',c='r',alpha=0.2)
ax_act_pd[1].set_xlabel("$I_{prox}$")
ax_act_pd[1].set_ylabel("$I_{dist}$")

ax_act_pd[0].set_title("First " + str(int(100.*t_wind/n_t_learn))+"% of learning phase")
ax_act_pd[1].set_title("Last " + str(int(100.*t_wind/n_t_learn))+"% of learning phase")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/act_pd_color_scatter_point_neuron_pc_defl.png",dpi=300)
###

###
fig_w, ax_w = plt.subplots(figsize=(5.,2.5))
ax_w.plot(w_prox_rec)
ax_w.set_xlabel("#t")
ax_w.set_ylabel("$w_{prox}$")

plt.tight_layout()
plt.savefig("../notes/presentation/figures/act_pd_w_point_neuron_pc_defl.png",dpi=300)
###


###
fig_I, ax_I = plt.subplots(2,1,figsize=(5,4))
t_wind = int(n_t_learn*0.01)

ax_I[0].plot(I_d_rec[:t_wind])
ax_I[0].plot(I_p_rec[:t_wind])

ax_I[0].set_xlabel("#t")
ax_I[0].set_ylabel("$I_{prox}$, $I_{dist}$")
ax_I[0].set_title("First " + str(int(100.*t_wind/n_t_learn))+"% of learning phase")

ax_I[1].plot(t_ax[-t_wind:],I_d_rec[-t_wind:])
ax_I[1].plot(t_ax[-t_wind:],I_p_rec[-t_wind:])

ax_I[1].set_xlabel("#t")
ax_I[1].set_ylabel("$I_{prox}$, $I_{dist}$")
ax_I[1].set_title("Last " + str(int(100.*t_wind/n_t_learn))+"% of learning phase")

plt.tight_layout()
plt.savefig("../notes/presentation/figures/I_pd_point_neuron_pc_defl.png",dpi=300)
###

###
t_wind = int(n_t_learn*0.005)
fig_X_p, ax_X_p = plt.subplots(figsize=(5,2.))
ax_X_p.plot(X_p[:t_wind])
ax_X_p.set_xlabel("#t")
ax_X_p.set_ylabel("$x_{prox}$")

plt.tight_layout()
plt.savefig("../notes/presentation/figures/X_p_point_neuron_pc_defl.png",dpi=300)
###

plt.show()

pdb.set_trace()

