#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm





def s(x):
	return np.tanh(x)

def act_pos(x):

	return (np.tanh(2.*x)+1.)/2.

def act_pd_2(p,d,gain,theta_p0,theta_p1,theta_d,alpha):

	return act_pos(gain*(p-theta_p1))*act_pos(gain*(d-theta_d)) + alpha*act_pos(gain*(p-theta_p0))*act_pos(-gain*(d-theta_d))


def input(t):

	return np.sin(np.pi*2.*t/8.)+np.sin(np.pi*2.*t/32.)
	#return (np.sin(np.pi*2.*t/8.)+np.sin(np.pi*2.*t/32.))/8. + 0.5

N = 300



mu_learn = 0.005

n_t = 3000000
n_t_rec = 300000
n_t_skip = int(n_t/n_t_rec)

x = np.zeros(N)
y = 0.




w_dist = 1.

alpha_pd = 0.05

gain_pd = 5.

w_xy = np.random.normal(0.,1.,(N))


cf = 0.1
g = 0.8
W = np.random.normal(0.,1.,(N,N))*(np.random.rand(N,N) <= cf)*g/((cf*N)**.5)
W[range(N),range(N)] = 0.

w_yx_total= 1.
w_yx_max = w_yx_total
w_yx_min = 0.000001

w_yx = np.random.rand(N)

#w_yx = w_yx_total * w_yx/w_yx.sum()





x_mean = 0.5*np.ones(N)
mu_x_mean = 0.001

y_mean = 0.5
mu_y_mean = 0.001

I_p_mean = 0.5
I_d_mean = 0.5
mu_I_p_mean = 0.001
mu_I_d_mean = 0.001


w_yx_rec = np.ndarray((n_t_rec,N))
dw_yx_rec = np.ndarray((n_t))

Err_rec = np.ndarray((n_t))

#x_rec = np.ndarray((n_t,N))
y_rec = np.ndarray((n_t))

#x_mean_rec = np.ndarray((n_t,N))
y_mean_rec = np.ndarray((n_t))

I_p_rec = np.ndarray((n_t))
I_d_rec = np.ndarray((n_t))

I_p_mean_rec = np.ndarray((n_t))
I_d_mean_rec = np.ndarray((n_t))


for t in tqdm(range(n_t)):

	I_p = np.dot(x,w_yx)
	I_d = w_dist*input(t)

	th_p0 = I_p_mean#0.7*I_p_mean
	th_p1 = 0.1*I_p_mean
	th_d = 1.4*I_d_mean

	#I_p_x = np.dot(W,x) + w_xy*y

	#th_p0_x = I_p_x

	dw_yx = -mu_learn * (I_p-I_d)*x
	#dw_yx = mu_learn*(y-y_mean)*(x-x_mean)

	#x = act_pos(np.dot(W,x) + w_xy*y)
	x = s(np.dot(W,x) + w_xy*y)
	#x = act_pd_2(np.dot(W,x) + w_xy*y,0.,gain_pd,th_p0_x,0.,1.,alpha_pd)
	y = act_pd_2(I_p,I_d,gain_pd,th_p0,th_p1,th_d,alpha_pd)
	#y = act_pos(I_p+I_d)
	#y = s(I_p + I_d)
	#y = I_d
	w_yx += dw_yx

	x_mean += mu_x_mean*(-x_mean + x)
	y_mean += mu_y_mean*(-y_mean + y)

	I_p_mean += mu_I_p_mean*(-I_p_mean + I_p)
	I_d_mean += mu_I_d_mean*(-I_d_mean + I_d)

	#w_yx += mu_learn * (y-y_mean)*(x-x_mean)
	

	#w_yx = np.maximum(w_yx_min,w_yx)
	#w_yx = np.minimum(w_yx_max,w_yx)
	#w_yx = w_yx_total * w_yx/w_yx.sum()

	#x_rec[t,:] = x
	#x_mean_rec[t,:] = x_mean

	y_rec[t] = y
	y_mean_rec[t] = y_mean

	if t%n_t_skip == 0:
		w_yx_rec[int(t/n_t_skip),:] = w_yx
	dw_yx_rec[t] = np.linalg.norm(dw_yx)

	I_p_rec[t] =I_p
	I_d_rec[t] = I_d

	I_p_mean_rec[t] = I_p_mean
	I_d_mean_rec[t] = I_d_mean

	Err_rec[t] = np.abs(I_p-I_d)

input_sequ = input(np.array(range(n_t)))

np.save('./sim_data/echo_state_pd/y_rec',y_rec)
np.save('./sim_data/echo_state_pd/y_mean_rec',y_mean_rec)
np.save('./sim_data/echo_state_pd/w_yx_rec',w_yx_rec)
np.save('./sim_data/echo_state_pd/I_p_rec',I_p_rec)
np.save('./sim_data/echo_state_pd/I_d_rec',I_d_rec)
np.save('./sim_data/echo_state_pd/I_d_mean_rec',I_d_mean_rec)
np.save('./sim_data/echo_state_pd/I_p_mean_rec',I_p_mean_rec)
np.save('./sim_data/echo_state_pd/Err_rec',Err_rec)
np.save('./sim_data/echo_state_pd/input_sequ',input_sequ)
np.save('./sim_data/echo_state_pd/w_xy',w_xy)

t_ax = np.array(range(n_t))
t_ax_skip = np.linspace(0,n_t,n_t_rec)

fig_end,ax_end = plt.subplots(figsize=(6,3))

ax_end.plot(t_ax[-100:],I_d_rec[-100:],label="$I_d$")

ax_end.plot(t_ax[-100:],I_p_rec[-100:],label="$I_p$")

ax_end.legend()

ax_end.set_xlabel("time step")
ax_end.set_ylabel("$I_p / I_d$")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/echo_state_network_pd_act_grad_desc_end.png",dpi=300)

fig_start,ax_start = plt.subplots(figsize=(6,3))

ax_start.plot(t_ax[:100],I_d_rec[:100],label="$I_d$")

ax_start.plot(t_ax[:100],I_p_rec[:100],label="$I_p$")
ax_start.legend()

ax_start.set_xlabel("time step")
ax_start.set_ylabel("$I_p / I_d$")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/echo_state_network_pd_act_grad_desc_start.png",dpi=300)

fig_w,ax_w = plt.subplots(figsize=(6,3))

ax_w.plot(t_ax_skip[::100],w_yx_rec[::100,:])

ax_w.set_xlabel("time step")
ax_w.set_ylabel("$w_{yx}$")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/echo_state_network_pd_act_grad_desc_w_yx.png",dpi=300)


height_total = 4.2
width_total = 16.125

fig_poster = plt.figure(figsize=(width_total,height_total))

gsposter = gridspec.GridSpec(nrows=2,ncols=4,left=0.06,right=0.99,bottom=0.15,top=0.95,wspace=0.43,hspace=0.6) 


plt.show()


pdb.set_trace()