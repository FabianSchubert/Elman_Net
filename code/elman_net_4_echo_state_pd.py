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

	return (np.sin(np.pi*2.*t/8.)+np.sin(np.pi*2.*t/32.)+2.)/4.

N = 500



mu_learn = 0.0001

n_t = 100000

x = np.zeros(N)
y = 0.

cf = 0.1
g = 0.8
W = np.random.normal(0.,1.,(N,N))*(np.random.rand(N,N) <= cf)*g/(cf*N)**.5
W[range(N),range(N)] = 0.

w_yx_total= 1.
w_yx_max = w_yx_total
w_yx_min = 0.

w_yx = np.random.rand(N)

w_yx = w_yx_total * w_yx/w_yx.sum()


w_xy = np.random.normal(0.,1.,(N))*0.5

w_dist = 1.

alpha_pd = 0.05

gain_pd = 5.

x_mean = 0.5*np.ones(N)
mu_x_mean = 0.001

y_mean = 0.5
mu_y_mean = 0.001

I_p_mean = 0.5
I_d_mean = 0.5
mu_I_p_mean = 0.001
mu_I_d_mean = 0.001


w_yx_rec = np.ndarray((n_t,N))

Err_rec = np.ndarray((n_t))

x_rec = np.ndarray((n_t,N))
y_rec = np.ndarray((n_t))

x_mean_rec = np.ndarray((n_t,N))
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

	x = act_pos(np.dot(W,x) + w_xy*y)
	y = act_pd_2(I_p,I_d,gain_pd,th_p0,th_p1,th_d,alpha_pd)

	x_mean += mu_x_mean*(-x_mean + x)
	y_mean += mu_y_mean*(-y_mean + y)

	I_p_mean += mu_I_p_mean*(-I_p_mean + I_p)
	I_d_mean += mu_I_d_mean*(-I_d_mean + I_d)

	w_yx += mu_learn * (y-y_mean)*(x-x_mean)

	w_yx = np.maximum(w_yx_min,w_yx)
	w_yx = np.minimum(w_yx_max,w_yx)
	w_yx = w_yx_total * w_yx/w_yx.sum()

	x_rec[t,:] = x
	x_mean_rec[t,:] = x_mean

	y_rec[t] = y
	y_mean_rec[t] = y_mean

	w_yx_rec[t,:] = w_yx

	I_p_rec[t] =I_p
	I_d_rec[t] = I_d

	I_p_mean_rec[t] = I_p_mean
	I_d_mean_rec[t] = I_d_mean


input_sequ = input(np.array(range(n_t)))

t_ax = np.array(range(n_t))



pdb.set_trace()