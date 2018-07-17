#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm


N = 300

cf = 0.1
g = 0.8
W = np.random.normal(0.,1.,(N,N))*(np.random.rand(N,N) <= cf)*g/(cf*N)**.5
W[range(N),range(N)] = 0.

dW = np.zeros((N,N))

T = np.zeros(N)
dT = np.zeros(N)

w_ext = np.random.rand(N)*0.1

w_out = np.random.normal(0.,1.,(N))*1./N
dw_out = np.ndarray((N))


def s(x):
	return np.tanh(x)



def input(t):

	return np.sin(np.pi*2.*t/8.)+np.sin(np.pi*2.*t/32.)


mu_learn_w_out = 0.001

n_t = 100000

x = np.zeros(N)
y = 0.


x_rec = np.ndarray((n_t,N))
y_rec = np.ndarray((n_t))

w_out_rec = np.ndarray((n_t,N))

Err_rec = np.ndarray((n_t))


for t in tqdm(range(n_t)):

	x = s(np.dot(W,x) - T + w_ext*input(t-1))
	y = np.dot(w_out,x)

	dw_out = -x*(y-input(t))
	
	w_out += mu_learn_w_out * dw_out

	x_rec[t,:] = x
	y_rec[t] = y

	w_out_rec[t,:] = w_out 

	Err = np.abs(y-input(t))

	Err_rec[t] = Err


input_sequ = input(np.array(range(n_t)))

t_ax = np.array(range(n_t))

fig_err, ax_err = plt.subplots(figsize=(10,2.5))
ax_err.plot(t_ax[::50],Err_rec[::50]/input_sequ.std(),c='k')

ax_err.set_xlabel("time step")
ax_err.set_ylabel("RMS Prediction Error")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/echo_state_network_error_signal.png",dpi=300)

fig_sign,ax_sign = plt.subplots(1,3,figsize=(10,2.5))

ax_sign[0].plot(t_ax[:100],input_sequ[:100])
ax_sign[0].plot(t_ax[:100],y_rec[:100])
ax_sign[0].set_xlabel("time step")
ax_sign[0].set_ylabel("Output/Target")

t_start = int(n_t*0.01)
ax_sign[1].plot(t_ax,input_sequ)
ax_sign[1].plot(t_ax,y_rec)
ax_sign[1].set_xlim([t_start,t_start+100])
ax_sign[1].set_xlabel("time step")

t_start = int(n_t*0.9)
ax_sign[2].plot(t_ax,input_sequ)
ax_sign[2].plot(t_ax,y_rec)
ax_sign[2].set_xlim([t_start,t_start+100])
ax_sign[2].set_xlabel("time step")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/echo_state_network_learn_signal.png",dpi=300)

plt.show()

pdb.set_trace()