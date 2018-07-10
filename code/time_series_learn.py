#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import pdb

N = 200

T = 10000.
dt = 0.01
n_t = int(T/dt)

T_learn_end = 9500.
n_t_lear_end = int(T_learn_end/dt)

t_arr = np.array(range(n_t))*dt

g = 1.5

W_ee = np.random.normal(0.,1.,(N,N))*g/N**.5

w_re = np.random.rand(N)*0.1

w_er = np.random.normal(0.,1.,(N))*1.

mu_learn = .1

def act(x):

	return np.tanh(x)
	#return (np.tanh(2.*x)+1.)/2.

def t_sequ(t):

	return np.sin(t*0.2)+np.sin(t*0.6)


#x_rec = np.ndarray((n_t,N))

#r_rec = np.ndarray((n_t))

#w_re_rec = np.ndarray((n_t,N))

I_re_rec = np.ndarray((n_t))

x = np.random.normal(0.,.5,(N))

r = np.random.normal()

for t in tqdm(range(n_t)):

	I_ee = np.dot(W_ee,x)

	I_er = r*w_er

	I_re = np.dot(w_re,x)

	I_rext = t_sequ(t*dt)

	r += dt*(-r + act(I_re + I_rext))

	#r = act(I_re + I_rext)

	x += dt*(-x + act(I_ee + I_er))

	if t < n_t_lear_end: 
		w_re += dt*mu_learn*(I_rext-I_re)*x#(r*x - r**2*w_re)*1./(1.+0.5*(I_re-I_rext)**2)

	#x_rec[t,:] = x
	#r_rec[t] = r
	#w_re_rec[t,:] = w_re
	I_re_rec[t] = I_re 

'''
fig_x, ax_x = plt.subplots()

ax_x.plot(t_arr,x_rec)

ax_x.set_xlabel("t")
ax_x.set_ylabel("x")

fig_r, ax_r = plt.subplots()

ax_r.plot(t_arr,r_rec)

ax_r.set_xlabel("t")
ax_r.set_ylabel("r")

fig_w_re, ax_w_re = plt.subplots()

ax_w_re.plot(t_arr,w_re_rec)

ax_w_re.set_xlabel("t")
ax_w_re.set_ylabel("$w_{re}$")
'''
fig_I_r, ax_I_r = plt.subplots()

ax_I_r.plot(t_arr,I_re_rec)
ax_I_r.plot(t_arr,t_sequ(t_arr))

ax_I_r.set_xlabel("t")
ax_I_r.set_ylabel("$I_{re}$,$I_{rext}$")

plt.show()

pdb.set_trace()
