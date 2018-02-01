#!/usr/bin/python3

import sys
sys.path.append("../../../custom_modules/")
from plot_settings_mpl import *

import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm



def s(x):
	return np.tanh(x)
	#return (np.tanh(x/2.)+1.)/2.

def ds(x):
	return 1.-s(x)**2
	#return s(x)*(1.-s(x))


def update_layers(I,h,c,W_h,W_o,n_i,n_h,n_o):

	I_h = np.ones((n_i+n_h+1))
	I_o = np.ones((n_h+1))
	
	I_h[:n_i] = I
	I_h[n_i:n_i+n_h] = c

	I_o[:n_h] = h

	h_new = s(np.dot(W_h,I_h))
	c_new = h[:]
	o_new = s(np.dot(W_o,I_o))

	return (h_new,c_new,o_new)

def update_weights(I,h,o,f,W_h,W_o,n_t_bpp,n_i,n_h,n_o,mu_learn,mu_learn_out):

	H = np.ndarray((n_t_bpp,n_i+n_h+1))

	#####
	'''H[:,:n_i] = I
	H[0,n_i:n_i+n_h] = h[0,:]
	H[:,-1] = 1.

	for k in range(1,n_t_bpp):
		#pdb.set_trace()
		H[k,n_i:n_i+n_h] = s(np.dot(W_h,H[k-1,:]))
	o = s(np.dot(W_o,np.append(H[-1,n_i:n_i+n_h],1.)))'''
	#####

	H[:,:n_i] = I
	H[:,n_i:n_i+n_h] = h
	H[:,-1] = 1.

	dh = ds(H[:,n_i:n_i+n_h])
	do = ds(o)

	dW_h = np.zeros((n_h,n_i+n_h+1))
	dW_o = np.zeros((n_o,n_h + 1))

	#pdb.set_trace()

	for k in range(n_t_bpp):

		if k==0:

			#pdb.set_trace()

			dW_o = -mu_learn_out * np.outer(do*(o-f),np.append(h[-1,:],1))

			dE_h = np.dot(W_o.T[:n_h,:],do*(o-f))

		else:

			#pdb.set_trace()

			dW_h -= mu_learn * np.outer(dE_h*dh[-k,:],H[-k-1,:])

			dE_h = np.dot(W_h.T[n_i:n_i+n_h,:],dh[-k,:]*dE_h)
			#dE_I = np.dot(W_h.T[:n_i,:],dh[-k,:])
			#dE_b = np.dot(W_h.T[-1,:])
		


	#dW_h /= n_t_bpp
	#pdb.set_trace()

	return W_h + dW_h, W_o + dW_o



	

def main():

	n_i = 1
	n_h = 10
	n_o = 1

	n_t = 50000

	n_t_bpp = 5

	mu_learn = .05
	mu_learn_out = 0.025

	I = np.ndarray((n_t,n_i))
	
	for k in range(n_t):
		if k%3 == 2:
			I[k,0] = 1.*(I[k-1,0] != I[k-2,0])
		else:
			I[k,0] = int(np.random.rand()+.5)
	#I[:,0] = np.sin(np.array(range(n_t))*0.125*2.*np.pi)*0.25
	I = (I-.5)*.5
	#pdb.set_trace()


	#f = np.ndarray((n_t,n_o))
	#f[:,0] = np.sin(np.linspace(0.,50*2.*np.pi,n_t))*0.25+.5
	#f[:,1] = f[:,0]

	h = np.zeros(n_h)
	c = np.zeros(n_h)
	o = np.zeros(n_o)

	W_h = np.random.rand(n_h,n_i+n_h+1)*0.2 - .1 #external input + input from context unit + 1 input for "bias weight"

	W_o = np.random.rand(n_o,n_h + 1)*0.2 - .1 #input from hidden unit + 1 bias weight

	h_rec_bpp = np.zeros((n_t_bpp,n_h))

	h_rec = np.ndarray((n_t,n_h))
	o_rec = np.ndarray((n_t,n_o))

	E_rec = np.ndarray((n_t))
	
	W_h_rec = np.ndarray((n_t,n_h,n_i+n_h+1))
	W_o_rec = np.ndarray((n_t,n_o,n_h + 1))

	for t in tqdm(range(n_t)):

		h,c,o = update_layers(I[t,:],h,c,W_h,W_o,n_i,n_h,n_o)

		h_rec_bpp[:-1,:] = h_rec_bpp[1:,:]
		h_rec_bpp[-1,:] = c
		
		if t >= n_t_bpp+1 and t<n_t*0.75:
			W_h,W_o = update_weights(I[t-n_t_bpp+1:t+1,:],h_rec_bpp,o,I[t+1,:],W_h,W_o,n_t_bpp,n_i,n_h,n_o,mu_learn,mu_learn_out)

		h_rec[t,:] = h
		o_rec[t,:] = o
		
		W_h_rec[t,:,:] = W_h
		W_o_rec[t,:,:] = W_o

		if t<n_t-1:
			E_rec[t] = np.linalg.norm(o-I[t+1])**2

	fig, ax = plt.subplots(2,1)
	ax[0].plot(I[1:],label='Input')
	ax[0].plot(o_rec[:,0],'--',c='k',label="Predicted Output")
	
	ax[0].set_xlim([n_t-100,n_t])

	ax[0].legend()

	ax[1].plot(I[1:])
	ax[1].plot(o_rec,'--',c='k')
	
	ax[1].set_xlim([0,1000])

	ax[1].set_xlabel("Time Step")

	plt.show()

	#pdb.set_trace()

	for k in range(n_h):
		plt.plot(W_h_rec[:,k,:])

	plt.show()

	for k in range(n_o):
		plt.plot(W_o_rec[:,k,:])

	plt.show()
	pdb.set_trace()

if __name__ == "__main__":
	main()