#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter

from plot_setting import *

y = np.load('../sim_data/echo_state_pd/y_rec.npy')
y_mean = np.load('../sim_data/echo_state_pd/y_mean_rec.npy')
w_yx = np.load('../sim_data/echo_state_pd/w_yx_rec.npy')
I_p = np.load('../sim_data/echo_state_pd/I_p_rec.npy')
I_d = np.load('../sim_data/echo_state_pd/I_d_rec.npy')
I_p_mean = np.load('../sim_data/echo_state_pd/I_p_mean_rec.npy')
I_d_mean = np.load('../sim_data/echo_state_pd/I_d_mean_rec.npy')
Err = np.load('../sim_data/echo_state_pd/Err_rec.npy')
input_sequ = np.load('../sim_data/echo_state_pd/input_sequ.npy')

n_t = input_sequ.shape[0]
n_t_rec = w_yx.shape[0]

t_ax = np.array(range(n_t))
t_ax_skip = np.linspace(0,n_t,n_t_rec)

height_total = 6.5
width_total = 9

fig_poster,ax = plt.subplots(2,1,figsize=(width_total,height_total))

ax[0].plot(t_ax[:100],I_d[:100],label="$I_d$")

ax[0].plot(t_ax[:100],I_p[:100],label="$I_p$")

ax[0].set_xlabel("time step")
ax[0].set_ylabel("$I_p / I_d$")

ax[1].plot(t_ax[-100:],I_d[-100:],label="$I_d$")

ax[1].plot(t_ax[-100:],I_p[-100:],label="$I_p$")

ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#ax[1].ticklabel_format(axis='x', useMathText=True)	

ax[1].legend()

ax[1].set_xlabel("time step")
ax[1].set_ylabel("$I_p / I_d$")

plt.tight_layout()

imgformat = 'pdf'

poster_fig_folder = "/home/fschubert/work/repos/Poster_Bernstein/figures/"

fig_poster.savefig(poster_fig_folder + "fig4_right." + imgformat,dpi=300)

plt.show()