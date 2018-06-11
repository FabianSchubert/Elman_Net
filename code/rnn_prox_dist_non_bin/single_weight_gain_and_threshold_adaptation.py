#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import pdb

mu_w = 0.001
mu_a = 0.001
mu_b = 0.001



mu_y_pre = .5
std_y_pre = .1

y_target = 0.5
std_target = 0.1

n_t = 10000
n_trials = 50

rec = np.ndarray((n_trials,n_t,3))

def s(x,a,b):

	return (np.tanh(a*(x-b)/2.)+1.)/2.

def dw(x,a,b,y_pre,y_pre_mean):

	G = 2. + a*x*(1.-2*s(x,a,b))
	H = 2.*s(x,a,b) - 1. + 2.*a*x*(1.-s(x,a,b))*s(x,a,b)
	return 2.*a*G*H*(y_pre - y_pre_mean)

def da(x,a,b):

	return std_target**2 - (s(x,a,b) - y_target)**2

def db(x,a,b):

	return s(x,a,b) - y_target



def F(phi):

	r = np.ndarray((3))

	y_pre = np.random.normal(mu_y_pre,std_y_pre)

	x = y_pre*phi[0]
	a = phi[1]
	b = phi[2]

	r[0] = mu_w * dw(x,a,b,y_pre,mu_y_pre)
	r[1] = mu_a * da(x,a,b)
	r[2] = mu_b * db(x,a,b)

	return r


for k in tqdm(range(n_trials)):

	v = (np.random.rand(3) - .5)*2.

	for t in range(n_t):

		v += F(v)

		rec[k,t,:] = v


pdb.set_trace()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for k in range(n_trials):
	plt.plot(rec[k,:,0],rec[k,:,1],rec[k,:,2])

ax.set_xlabel("w")
ax.set_ylabel("a")
ax.set_zlabel("b")

plt.show()

pdb.set_trace()