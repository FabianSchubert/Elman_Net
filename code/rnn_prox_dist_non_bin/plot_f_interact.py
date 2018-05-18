#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

m_0 = 0.5
a_m = 0.5
theta_m = 0.5
s_m = 0.05

a_t = 0.33
theta_t = 0.5
s_t = 0.1

theta_p = 0.5
s_p = 0.05

def sigm(x):
	return (np.tanh(x/2.)+1.)/2.

def f(xp,xd):
	M = m_0 + a_m*sigm((xd-theta_m)/s_m)
	T = a_t*sigm((xd-theta_t)/s_t)
	return M*sigm((xp+T-theta_p)/s_p)


fig = plt.figure(figsize=(11,5))
gs = gridspec.GridSpec(10,2)
ax = [plt.subplot(gs[:,0])]
for k in range(9):
	ax.append(plt.subplot(gs[k,1]))


#ax[0].set_position([0.05, 0.05, 0.4, 0.9])
#ax[1].set_position([0.05, 0.05, 0.4, 0.9])

x = np.linspace(0.,1.,200)
y = np.linspace(0.,1.,200)

X,Y = np.meshgrid(x,y)

mesh = ax[0].pcolormesh(x,y,f(X,Y))

s_m_0 = Slider(ax[1], 'm_0', 0.0, 1.0, valinit=m_0)
s_a_m = Slider(ax[2], 'a_m', 0.0, 1.0, valinit=a_m)
s_theta_m = Slider(ax[3], 'theta_m', 0.0, 1.0, valinit=theta_m)
s_s_m = Slider(ax[4], 's_m', 0.0, 0.2, valinit=s_m)

s_a_t = Slider(ax[5], 'a_t', 0.0, 1.0, valinit=a_t)
s_theta_t = Slider(ax[6], 'theta_t', 0.0, 1.0, valinit=theta_t)
s_s_t = Slider(ax[7], 's_t', 0.0, 0.2, valinit=s_t)

s_theta_p = Slider(ax[8], 'theta_p', 0.0, 1.0, valinit=theta_p)
s_s_p = Slider(ax[9], 's_p', 0.0, 0.2, valinit=s_p)

def update(val):
	global m_0
	global a_m
	global theta_m
	global s_m

	global a_t
	global theta_t
	global s_t

	global theta_p
	global s_p

	m_0 = s_m_0.val
	a_m = s_a_m.val
	theta_m = s_theta_m.val
	s_m = s_s_m.val

	a_t = s_a_t.val
	theta_t = s_theta_t.val
	s_t = s_s_t.val

	theta_p = s_theta_p.val
	s_p = s_s_p.val
	#print(a_t)
	mesh.set_array(f(X,Y)[:-1,:-1].ravel())
	fig.canvas.draw_idle()

s_m_0.on_changed(update)
s_a_m.on_changed(update)
s_theta_m.on_changed(update)
s_s_m.on_changed(update)

s_a_t.on_changed(update)
s_theta_t.on_changed(update)
s_s_t.on_changed(update)

s_theta_p.on_changed(update)
s_s_p.on_changed(update)

#plt.tight_layout()

plt.show()

 
