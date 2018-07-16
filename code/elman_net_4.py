#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm


N = 50

W = np.random.normal(0.,1.,(N,N))*0.5/N**0.5
dW = np.zeros((N,N))

T = np.zeros(N)
dT = np.zeros(N)

w_ext = np.random.rand(N)*0.1
dw_ext = np.zeros(N)

w_out = np.random.normal(0.,1.,(N))*1./N
dw_out = np.zeros(N)

n_t_back = 10

x_mem = np.zeros((n_t_back+1,N))

dx_mem = np.zeros((n_t_back+1,N))



def s(x):
	return np.tanh(x)



def input(t):

	return np.sin(np.pi*2.*t/8.)+np.sin(np.pi*2.*t/32.)


mu_learn_W = 0.1
mu_learn_T = 0.001
mu_learn_w_ext = 0.001
mu_learn_w_out = 0.001

n_t = 10000

x = np.zeros(N)
y = 0.





x_rec = np.ndarray((n_t,N))
y_rec = np.ndarray((n_t))

W_rec = np.ndarray((n_t,N,N))
T_rec = np.ndarray((n_t,N))
w_ext_rec = np.ndarray((n_t,N))
w_out_rec = np.ndarray((n_t,N))

Err_rec = np.ndarray((n_t))


for t in tqdm(range(n_t)):

	x = s(np.dot(W,x) - T + w_ext*input(t-1))
	y = np.dot(w_out,x)

	x_mem[1:,:] = x_mem[:-1,:]
	x_mem[0,:] = x

	dx_mem = (1.+x_mem)*(1.-x_mem)

	dw_out = -x*(y-input(t))

	dEdx = w_out*(y-input(t))

	dW = np.zeros((N,N))
	dT = np.zeros(N)
	dw_ext = np.zeros(N)

	for k in range(n_t_back):

		dW -= np.outer(dx_mem[k,:]*dEdx,x_mem[k+1,:])
		dT -= -dx_mem[k,:]*dEdx
		dw_ext -= input(t-1-k)*dx_mem[k,:]*dEdx

		dEdx = np.dot(W.T,dx_mem[k,:]*dEdx)


	W += mu_learn_W * dW/n_t_back
	T += mu_learn_T * dT/n_t_back
	w_ext += mu_learn_w_ext * dw_ext/n_t_back
	w_out += mu_learn_w_out * dw_out


	x_rec[t,:] = x
	y_rec[t] = y

	W_rec[t,:,:] = W
	T_rec[t,:] = T
	w_ext_rec[t,:] = w_ext
	w_out_rec[t,:] = w_out 

	Err_rec[t] = np.abs(y-input(t))


input_sequ = input(np.array(range(n_t)))

t_ax = np.array(range(n_t))

fig_err, ax_err = plt.subplots(figsize=(10,2.5))
ax_err.plot(t_ax[::50],Err_rec[::50]/input_sequ.std(),c='k')

ax_err.set_xlabel("time step")
ax_err.set_ylabel("RMS Prediction Error")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/backpropagation_through_time_error_signal.png",dpi=300)

fig_sign,ax_sign = plt.subplots(1,3,figsize=(10,2.5))

ax_sign[0].plot(t_ax[:100],input_sequ[:100])
ax_sign[0].plot(t_ax[:100],y_rec[:100])
ax_sign[0].set_xlabel("time step")
ax_sign[0].set_ylabel("Output/Target")

t_start = int(n_t*0.125)
ax_sign[1].plot(t_ax,input_sequ)
ax_sign[1].plot(t_ax,y_rec)
ax_sign[1].set_xlim([t_start,t_start+100])
ax_sign[1].set_xlabel("time step")

t_start = int(n_t*0.5)
ax_sign[2].plot(t_ax,input_sequ)
ax_sign[2].plot(t_ax,y_rec)
ax_sign[2].set_xlim([t_start,t_start+100])
ax_sign[2].set_xlabel("time step")

plt.tight_layout()

plt.savefig("../notes/presentation/figures/backpropagation_through_time_learn_signal.png",dpi=300)

plt.show()

pdb.set_trace()