#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from styles import *


def theta(y,m,s):

	return 1.-2.*y + y*(1.-y)*(m-y)/s**2

def db(y,m,s):

	return theta(y,m,s)

def da(y,m,s):

	return 1.- np.log(1./y - 1.)*theta(y,m,s)


y = np.linspace(0.0001,0.999,2000)


murange = np.array([0.125,0.25,0.5,0.75,0.875])#np.linspace(0.25,0.75,3)

srange = np.array([0.1,0.2,0.3])#np.linspace(0.05,0.1,0.15,3)

fig = plt.figure(figsize=(textwidth,7))

gs = gridspec.GridSpec(3,2,height_ratios=[5,5,1])
ax = [[],[]]
ax[0].append(plt.subplot(gs[0,0]))
ax[0].append(plt.subplot(gs[0,1]))
ax[1].append(plt.subplot(gs[1,0]))
ax[1].append(plt.subplot(gs[1,1]))

ax = np.array(ax)

ax_col_db = plt.subplot(gs[2,0])
ax_col_da = plt.subplot(gs[2,1])

ax_col_da.tick_params(bottom=False,left=False)
ax_col_db.tick_params(bottom=False,left=False)

ax_col_da.axis('off')
ax_col_db.axis('off')

ax_col_db.text(0,1,"$\\mu_t :$")
murange_str = ""

for m in murange:
	murange_str = murange_str + str(m) + "\n"

ax_col_db.text(0.2,-.8,murange_str)

rects = []




captionfont = fontdict={'fontsize':10,'fontweight':'bold'}
ax[0,0].set_title("A",loc="left",fontdict=captionfont)
ax[1,0].set_title("B",loc="left",fontdict=captionfont)
ax[0,1].set_title("C",loc="left",fontdict=captionfont)
ax[1,1].set_title("D",loc="left",fontdict=captionfont)

for k in range(murange.shape[0]):

	m = murange[k]

	col = (m-murange.min())/(murange.max()-murange.min())

	cmap = (0.,0.5+0.5*col,1.-0.5*col)

	db1 = db(y,m,0.05)
	#db1 /= (db1.max()-db1.min())
	db2 = db(y,m,0.2)
	#db2 /= (db2.max()-db2.min())



	ax[0,0].plot(y,db1,c=cmap)

	ax[1,0].plot(y,db2,c=cmap)

	ax[0,0].scatter([m],[0],c=cmap,s=10,linewidths=1,edgecolors='k',zorder=3)

	ax[1,0].scatter([m],[0],c=cmap,s=10,linewidths=1,edgecolors='k',zorder=3)

	rects.append(mpatches.Rectangle((0.,1.-k*1./murange.shape[0]),0.1,0.05,ec="none"))

	rects[-1].set_color(cmap)



for s in srange:

	col = (s-srange.min())/(srange.max()-srange.min())

	cmap = (0.6+0.4*col,0.7*col,0)

	da1 = da(y,0.5,s)
	#da1 /= (da1.max()-da1.min())
	da2 = da(y,0.8,s)
	#da2 /= (da2.max()-da2.min())

	ax[0,1].plot(y,da1,c=cmap)

	ax[1,1].plot(y,da2,c=cmap)

	ax[0,1].plot([0.5+s,0.5+s],[-1,1],c=cmap)
	ax[0,1].plot([0.5-s,0.5-s],[-1,1],c=cmap)

	ax[1,1].plot([0.8+s,0.8+s],[-2,2],c=cmap)
	ax[1,1].plot([0.8-s,0.8-s],[-2,2],c=cmap)



ax[0,1].scatter([0.5],[0],c=(0.6+0.4*col,0.7*col,0),s=10,linewidths=1,edgecolors='k',zorder=3)

ax[1,1].scatter([0.8],[0],c=(0.6+0.4*col,0.7*col,0),s=10,linewidths=1,edgecolors='k',zorder=3)


rectcoll = PatchCollection(rects)

ax_col_db.add_collection(rectcoll)


for k in range(2):
	for l in range(2):
		ax[k,l].grid()
		ax[k,l].set_xlim([0.,1.])
		if k==1:
			ax[k,l].set_xlabel("y")

plt.tight_layout()

plt.savefig("../figures/gain_thresh_dyn.png",dpi=400)

plt.show()