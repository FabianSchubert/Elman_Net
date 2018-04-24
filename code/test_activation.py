#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pdb


def sigma(x):

	return (np.tanh(x/2.)+1.)/2.



def s_pd(x_p,x_d,norm_x_p,norm_x_d,T_contr):
	x_p_norm = x_p/norm_x_p
	x_d_norm = x_d/norm_x_d

	M = a1 + a2*sigma((x_d_norm-a3)/a4)
	T = b1 + b2*sigma((x_d_norm-b3)/b4)
	f = (M*sigma((x_p_norm-T)/c))**(1./T_contr)
	return f

# Parameters
a1 = 0.5
a2 = 0.5
a3 = 0.36
a4 = 0.05
b1 = 0.1
b2 = 0.5
b3 = 0.3
b4 = -0.063
c = 0.003

x_p = np.linspace(0.,1.,500)
x_d = np.linspace(0.,1.,500)

X_P,X_D = np.meshgrid(x_p,x_d)

plt.pcolormesh(x_p,x_d,s_pd(X_P,X_D,1.,1.,.01))
plt.colorbar()
plt.show()

pdb.set_trace()
