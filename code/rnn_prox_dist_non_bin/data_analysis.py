#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle


def load_data(path_x="./data/x_e_rec",path_W="./data/W_rec",path_I="./data/I_ee_rec",path_p="./data/parameters.p"):
	x = np.load("./data/x_e_rec.npy")
	W = np.load("./data/W_rec.npy")
	I = np.load("./data/I_ee_rec.npy")
	with open("./data/parameters.p","rb") as file:
		p = pickle.load(file)
	return x,W,I,p

def h(x_post_pot,x_pre,a):

	f = (x_post_pot*np.tanh(a)/a - np.tanh(x_post_pot))/a**3

	return np.outer(f,x_pre)


def plot_x():
	fig = plt.figure()
	t = np.linspace(0,p["n_t"],p["n_t_rec"])
	plt.plot(t,x,c='k',alpha=0.01)
	plt.xlabel("#t")
	plt.ylabel("$x_e$")
	plt.show()

def animate_mem_dist():
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




x,W,I,p = load_data()