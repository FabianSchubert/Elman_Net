#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle

import ast

import pdb

def load_data(path):
	dat = np.load(path)
	p = dat['param_dict'].tolist()
	x = dat['x_e_rec']
	x_mean = dat['x_e_mean_rec']
	x_ext = dat['x_ext_rec']
	x_ext_mean = dat['x_ext_mean_rec']
	store_w = p["store_w"]
	store_w_ext = p["store_w_ext"]
	W = dat['W_rec']
	W_eext = dat['W_eext_rec']
	gain = dat['gain_rec']
	thresh = dat['thresh_rec']
	I_ee = dat['I_ee_rec']
	I_eext = dat['I_eext_rec']
	
	return {"x":x,"x_mean":x_mean,"x_ext":x_ext,"x_ext_mean":x_ext_mean,"W":W,"W_eext":W_eext,"gain":gain,"thresh":thresh,"I_ee":I_ee,"I_eext":I_eext,"p":p}
	

def h(x_post_pot,x_pre,a):

	f = (x_post_pot*np.tanh(a)/a - np.tanh(x_post_pot))/a**3

	return np.outer(f,x_pre)

def act(x,gain):

	return (np.tanh(gain*x/2.)+1.)/2.

def d_act(x,gain):
	return gain*act(x,gain)*(1.-act(x,gain))


def plot_x():
	fig = plt.figure()
	t = np.linspace(0,p["n_t"],p["n_t_rec"])
	plt.plot(t,x,c='k',alpha=0.01)
	plt.xlabel("#t")
	plt.ylabel("$x_e$")
	plt.show()

def animate_mem_dist(I,p):
	fig, ax = plt.subplots()
	plt.subplots_adjust(left=0.25, bottom=0.25)

	hbins = np.linspace(-2.,2.,50)
	hist = np.histogram(I[0,:],bins=hbins,normed=True)
	step, = plt.step(hbins[1:],hist[0])

	x_space = np.linspace(-2.,2.,1000)
	l, = plt.plot(x_space,h(x_space,1.,p["a_hebb"])[:,0])

	plt.grid()
	plt.axis([hbins[0], hbins[-1], -1, 1])
	
	axtime = plt.axes([0.25, 0.1, 0.65, 0.03])
	stime = Slider(axtime,"Time Step",0,p["n_t"],valinit=0)

	def update(val):
		t = int(stime.val/p["n_t_skip"])
		hist = np.histogram(I[t,:],bins=hbins,normed=True)
		step.set_ydata(hist[0])
		fig.canvas.draw_idle()

	stime.on_changed(update)
	

	plt.show()




#x,W,I,p = load_data()
