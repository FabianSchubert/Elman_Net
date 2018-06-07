#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a_t = 0.33
theta_t = 0.5
s_t = 0.1

theta_p = 0.5

def sigm(x):
	return (np.tanh(x/2.)+1.)/2.

def T(x):
	return a_t*sigm((x - theta_t)/s_t);

Y = pd.read_csv("data_act.csv")
W = pd.read_csv("data_weights.csv")

X=pd.DataFrame(Y.values[:,:2]*W.values, columns=["x_p","x_d"], index=Y.index)

fig_act_weights, ax_act_weights = plt.subplots(2,1,figsize=(10,6))

Y.plot(ax=ax_act_weights[0])
W.plot(ax=ax_act_weights[1])

fig_scatter_act, ax_scatter_act = plt.subplots(1,1,figsize=(5,5))

scat = ax_scatter_act.scatter(X["x_p"][5000:],X["x_d"][5000:],c=Y["y_post"][5000:],lw=0,s=10)

ax_scatter_act.set_xlabel("$X_{p}$")
ax_scatter_act.set_ylabel("$X_{d}$")

plt.colorbar(mappable=scat, ax=ax_scatter_act)

plt.show()

plt.plot(X["x_p"] + T(X["x_d"]) - theta_p,label="X_p + T(X_d) - theta_p")
plt.plot(X["x_d"],label="X_d")
plt.ylabel("X")
plt.xlabel("t")
plt.legend()
plt.show()

import pdb
pdb.set_trace()