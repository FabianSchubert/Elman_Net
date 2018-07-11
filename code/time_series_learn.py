#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import pdb

N = 500

T = 15000.
dt = 0.1
n_t = int(T/dt)

T_learn_end = 9500.
n_t_learn_end = int(T_learn_end/dt)

t_arr = np.array(range(n_t))*dt

p_ee = 0.1

g = 3.

W_ee = np.random.normal(0.,g/(p_ee*N)**.5,(N,N))*(np.random.rand(N,N) <= p_ee)

w_re = (np.random.rand(N)-0.5)*0.1

w_er = 2.*(np.random.rand(N)-0.5)*2.

mu_learn = .2
def act(x):

	#return np.tanh(x)
	return (np.tanh(2.*x)+1.)/2.#

def act_p(x):
	return (np.tanh(2.*x)+1.)/2.


def act_pd(p,d,alpha):

	return (alpha + (1.-alpha)*act_p(d))*act(p)

def t_sequ(t):

	return (np.sin(t*0.2)+np.sin(t*0.6))/2.


#x_rec = np.ndarray((n_t,N))

#r_rec = np.ndarray((n_t))

w_re_rec = np.ndarray((n_t,N))

dw_re_rec = np.ndarray((n_t))

err_rec = np.ndarray((n_t))

I_re_rec = np.ndarray((n_t))

x = np.random.normal(0.,.5,(N))

r = np.random.normal()



for t in tqdm(range(n_t)):

	I_ee = np.dot(W_ee,x)

	I_er = r*w_er

	I_re = np.dot(w_re,x)

	if t < n_t_learn_end:

		I_rext = t_sequ(t*dt)

		#dw_re = mu_learn*r*x
		dw_re = mu_learn*(I_rext-I_re)*x*np.abs(I_rext-I_re)**(3./2.)
		w_re += dt*dw_re
		#w_re = np.maximum(0.0001,w_re)
		#w_re/= w_re.sum()
		
	else:
		I_rext = 0.
		dw_re = 0.
		

	r += dt*(-r + act(I_re))
	#r += dt*(-r + act_pd(I_re,I_rext,0.5))

	#r = act(I_re)
	#r = I_re

	#r += dt*(-r + act_pd(I_re - T_r, I_rext - T_r,0.25))

	#r = act(I_re + I_rext)

	x += dt*(-x + act(I_ee + I_er))
	
	#x_rec[t,:] = x
	#r_rec[t] = r
	w_re_rec[t,:] = w_re
	I_re_rec[t] = I_re

	dw_re_rec[t] = np.linalg.norm(dw_re)

	err_rec[t] = np.abs((t_sequ(t*dt)-I_re))

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

fig_dw_re, ax_dw_re = plt.subplots()

ax_dw_re.plot(t_arr,dw_re_rec)

ax_dw_re.set_xlabel("t")
ax_dw_re.set_ylabel("$|\\dot{\\mathrm{w}_{re}}|$")

fig_err, ax_err = plt.subplots()

ax_err.plot(t_arr,err_rec)

ax_err.set_xlabel("t")
ax_err.set_ylabel("$|I_{re} - I_{rext}|$")

plt.show()

pdb.set_trace()
