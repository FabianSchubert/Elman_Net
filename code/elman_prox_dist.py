#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

def s(x):
	return (np.tanh(x/2.) + 1.)/2.

def ds(x):
	return s(x)*(1.-s(x))

def f_comp(x_p,x_d,a1,a2,a3,a4,b1,b2,b3,b4,c):
	M = a1 + a2*s((x_d-a3)/a4)
	T = b1 + b2*s((x_d-b3)/b4)
	f = M*s((x_p-T)/c)
	return f

def oja(W,y,x):
	return np.outer(y,x) - (W.T*(y**2)).T - W

def hebb_norm(W,y,x,mu,inp_norm):
	W_new = W + (np.outer(y,x)-np.outer(x,y))*mu
	W_new = inp_norm*(W_new.T/W_new.sum(axis=1)).T
	return W_new

#def bcm(W,y,x):


#def ds(x):
#	return 1. - s(x)**2




def update_weights(I_h,h,I_o,o,f,W_h,W_o,n_i,n_h,n_o,mu_learn,mu_learn_out):

	dh = ds(h)
	do = ds(o)
	
	dW_h = np.zeros((n_h,n_i+n_h+1))
	dW_o = np.zeros((n_o,n_h + 1))

	#pdb.set_trace()

	dW_o = -mu_learn_out * np.outer(do*(o-f),I_o)
	dE_h = np.dot(W_o.T[:n_h,:],do*(o-f))

	dW_h = -mu_learn * np.outer(dE_h*dh,I_h)

	return dW_h, dW_o



def main():

	n_i = 1
	n_h = 20
	n_o = 1

	n_t = 20000

	mu_learn = np.ones((n_h,n_i+n_h))*.001
	mu_learn_out = np.ones((n_o,n_h + 1))*.01
	#mu_learn_out[:,-1] = .001

	n_trials = 1
	n_error_rec = int(0.25*n_t)
	E_rec_trials = np.ndarray((n_trials,n_error_rec))

	
	### composite neuron parameters
	a1 = .5
	a2 = .5
	a3 = .36
	a4 = .05
	b1 = .1
	b2 = .5
	b3 = .3
	b4 = -.063
	C = .003
	###



	for l in tqdm(range(n_trials)):


		h = np.zeros(n_h)
		c = np.zeros(n_h)
		o = np.zeros(n_o)

		W_h = np.random.rand(n_h,n_i+n_h)
		#W_o = np.random.rand(n_o,n_h+1)-.5
		W_o = np.ones((n_o,n_h+1))#np.random.rand(n_o,n_h + 1)*1. - .5 #input from hidden unit + 1 bias weight
		W_o[:,-1] = 0.


		dW_h = np.zeros((n_h,n_i+n_h+1))
		dW_o = np.zeros((n_o,n_h+1))

		I = np.ndarray((n_t,n_i))
		
		'''
		for k in range(n_t):
			if k%3 == 2:
				I[k,0] = 1.*(I[k-1,0] != I[k-2,0])
			else:
				I[k,0] = int(np.random.rand()+.5)
		
		#I = I*2.
		'''
		I[:,0] = (np.sin(np.array(range(n_t))*0.25*2.*np.pi)+1.)/2.
		
		
		I_h = np.ones((n_i+n_h))
		I_o = np.ones((n_h+1))

		o_rec = np.ndarray((n_t,n_o))
		h_rec = np.ndarray((n_t,n_h))
		E_rec = np.ndarray((n_t-1))

		W_h_rec = np.ndarray((n_t,n_h,n_i+n_h))
		W_o_rec = np.ndarray((n_t,n_o,n_h+1))

		I_p_rec = np.ndarray((n_t,n_h))
		I_d_rec = np.ndarray((n_t,n_h))

		I_p_bias = 0.
		I_d_bias = 0.

		I_p_bias_rec = np.ndarray((n_t))
		I_d_bias_rec = np.ndarray((n_t))

		for t in tqdm(range(n_t)):

			I_p = np.dot(W_h[:,:n_i],I[t,:])-I_p_bias
			I_d = np.dot(W_h[:,n_i:n_i+n_h],c)-I_d_bias

			I_h[:n_i] = I[t,:]
			I_h[n_i:] = c

			h = f_comp(I_p,I_d,a1,a2,a3,a4,b1,b2,b3,b4,C)

			I_p_bias += 0.005*(h.mean()-.5)
			I_d_bias += 0.005*(h.mean()-.5)

			c[:] = h


			I_o[:n_h] = h[:]

			o = s(np.dot(W_o,I_o))

			
			
			if t < 0.75*n_t:
				#dW_h = mu_learn*oja(W_h,h,I_h)

				dW_o = update_weights(I_h,h,I_o,o,I[t+1,:],W_h,W_o,n_i,n_h,n_o,mu_learn,mu_learn_out)[1]

				W_h = hebb_norm(W_h,h,I_h,mu_learn,0.5)
				W_o += dW_o
			
			c[:] = h[:]


			if t<n_t-1:
				E_rec[t] = np.linalg.norm(o-I[t+1])**2

			o_rec[t,:] = o
			h_rec[t,:] = h
			W_h_rec[t,:,:] = W_h
			W_o_rec[t,:,:] = W_o

			I_p_rec[t,:] = I_p
			I_d_rec[t,:] = I_d
			I_p_bias_rec[t] = I_p_bias
			I_d_bias_rec[t] = I_d_bias

		E_rec_trials[l,:] = E_rec[-n_error_rec:]
		

	pdb.set_trace()
	
	
	fig1, ax1 = plt.subplots(2,1)
	ax1[0].plot(I[1:],label='Input')
	ax1[0].plot(o_rec[:,0],'--',c='k',label="Predicted Output")
	
	ax1[0].set_xlim([n_t-100,n_t])

	ax1[0].legend()

	ax1[1].plot(I[1:])
	ax1[1].plot(o_rec,'--',c='k')
	
	ax1[1].set_xlim([0,1000])

	ax1[1].set_xlabel("Time Step")
	
	
	fig2,ax2 = plt.subplots(1,1)
	for k in range(n_h):
		ax2.plot(W_h_rec[:,k,:],c='k',alpha=0.8)
	ax2.set_xlabel("Time Step")
	ax2.set_ylabel("W_h")
	
	fig3, ax3 = plt.subplots(1,1)
	for k in range(n_o):
		ax3.plot(W_o_rec[:,k,:],c='k',alpha=0.8)
	ax3.set_xlabel("Time Step")
	ax3.set_ylabel("W_o")

	plt.show()

	pdb.set_trace()


if __name__ == "__main__":
	main()




