#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

import pdb

y = np.linspace(0.00001,0.9999,2000)

def theta(y,m,s):

	return 1.-2.*y + y*(1.-y)*(m-y)/s**2

def db(y,m,s):

	return theta(y,m,s)

def da(y,m,s):

	return 1.- np.log(1./y - 1.)*theta(y,m,s)


def mu(y,s):

	return y + s**2*(2*y-1.)/(y*(1.-y))

def sigm(y,m):
	return 1./np.sqrt((y*(1.-y)*(m-y)*np.log(1./y-1.))**-1. + (2*y-1)/(y*(1.-y)*(m-y)))


srange = np.linspace(0.01,0.5,20)
murange = np.linspace(0.0001,.999,20)

y0_a = np.ndarray((murange.shape[0],srange.shape[0]))
y0_b = np.ndarray((murange.shape[0],srange.shape[0]))
'''
for k in range(srange.shape[0]):
	plt.plot(mu(y,srange[k]),y,c=(srange[k]/srange.max(),0,1.-srange[k]/srange.max()))

for k in range(murange.shape[0]):
	
	plt.plot(sigm(y,murange[k]),y,c=(0,murange[k]/murange.max(),0))
	plt.plot(-sigm(y,murange[k]),y,c=(0,murange[k]/murange.max(),0))

plt.show()
'''
for k in range(murange.shape[0]):

	for l in range(srange.shape[0]):
		f = lambda x: da(x,murange[k],srange[l])
		sol = fsolve(f,0.5,full_output=True)
		if sol[2]:
			y0_a[k,l] = sol[0]
		else:
			y0_a[k,l] = np.nan

		f = lambda x: db(x,murange[k],srange[l])
		sol = fsolve(f,0.5,full_output=True)
		if sol[2]:
			y0_b[k,l] = sol[0]
		else:
			y0_b[k,l] = np.nan


pdb.set_trace()
