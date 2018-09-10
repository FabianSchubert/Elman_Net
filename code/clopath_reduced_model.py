#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import pdb

## Parameters

dt = 0.0001 # simulation time step, secs

T = .5 # simulation time, secs


g_L = 40.*10.**(-3.) # leak conductance, mS
E_L = -69. # resting potential, mV
C = 281.*10.**(-6.) # membrane capacitance, mF
τ_ampa = 0.002 # AMPA time constant, secs
E_ampa = 0. # AMPA reversal potential, mV
τ_nmda = 0.05 # NMDA time constant, secs
E_nmda = 0. # NMDA reversal potentia, mV
g_ampa = 100.*10.**(-3.) # maximal AMPA conductance, mS
g_nmda = 50.*10.**(-3.) # NMDA conductance, mS
θ = 20. # spiking threshold, mV
Δ_T = 2. # slope factor, mV
V_Trest = -50.4 # adaptive threshold rest value, mV
V_Tmax = -30.4 # adaptive threshold max value, mV
τ_VT = 0.05 # adaptive threshold time constant, secs
V_reset = -55. # voltage reset, mV
τ_neg = 35. # low-pass filter time constant 1, ms
τ_pos = 35. # low-pass filter time constant 2, ms
θ_neg = -69. # depolarisation threshold for plasticity, mV
θ_pos = -15. # depolarisation threshold for potentiation, mV
Δ_spike = .001 # spike width, secs
V_spike = 30. # spike plateau potential, mV
x_reset = 1. # spike trace reset value
τ_x = 15. # spike trace time constant, ms
A_ltd = 5.*10.**(-4.) # depression amplitude, mV^-1
A_ltp = 15.*10.**(-4.) # potentiation amplitude, mV^-1

τ_noise = 0.02 # membrane noise time constant, secs
σ_noise = 0.#35.*10.**(-3.) # membrane noise std dev, mA 



w_prox = 1.
w_dist = 1.

n_comp = 4

n_mult_single_prox = 10.
n_mult_connect_prox = 10.

'''
comp 0: soma
comp 1: single proximal
comp 2: connecting proximal
comp 3: distal
'''
labels = ["soma","single proximal","connecting proximal","distal"]

g_axial_soma_som_prox_neg = 50.*10.**(-3.)
g_axial_proximal_som_prox_neg = 1250.*10.**(-3.)
g_axial_proximal_som_prox_pos = 2500.*10.**(-3.)
g_axial_proximal_prox_dist_neg = 225.*10.**(-3.)
g_axial_proximal_prox_dist_pos = 1500.*10.**(-3.)
g_axial_dist_dist_prox_pos = 1500.*10.**(-3.)
g_axial_dist_dist_prox_neg = 225.*10.**(-3.)

##

n_sweep_rate_prox = 10
n_sweep_rate_dist = 20

rate_prox_sweep = np.linspace(0.,100.,n_sweep_rate_prox)
rate_dist_sweep = np.linspace(0.,350.,n_sweep_rate_dist)

output_rate = np.ndarray((n_sweep_rate_prox,n_sweep_rate_dist))

for i in tqdm(range(n_sweep_rate_prox)):
	for j in range(n_sweep_rate_dist):

		r_prox = rate_prox_sweep[i] # input spike rate proximal, spikes/sec
		r_dist = rate_dist_sweep[j] # input spike rate distal, spikes/sec

		g_dyn_ampa_prox = 0.
		g_dyn_nmda_prox = 0.

		g_dyn_ampa_dist = 0.
		g_dyn_nmda_dist = 0.


		n_t = int(T/dt)
		T = n_t*dt

		u_rec = np.ndarray((n_t,n_comp))
		V_T_rec = np.ndarray((n_t))

		spike_list = [[],[],[],[]]

		u = V_reset*np.ones(n_comp)

		V_T = V_Trest

		spiking_state = 0.*np.ones(n_comp) # are we in the spike peak?
		t_last_spike = 0.*np.ones(n_comp) # time since last spike onset

		I_noise = 0.*np.ones(n_comp)

		I_axial = 0.*np.ones(n_comp)

		I_ext = 1.*np.ones(n_comp)

		I_ext[0] = 0.
		I_ext[1] = 0.
		I_ext[2] = 0.
		I_ext[3] = 0.

		I_exp = np.zeros(n_comp)

		for t in tqdm(range(n_t)):

			I_L = -g_L*(u - E_L)

			I_exp[0] = g_L * Δ_T * np.exp((u[0] - V_T)/Δ_T)

			I_ampa_prox = -g_dyn_ampa_prox * (u[1] - E_ampa)
			I_nmda_prox = -g_dyn_nmda_prox * (u[1] - E_nmda)

			I_ampa_dist = -g_dyn_ampa_dist * (u[3] - E_ampa)
			I_nmda_dist = -g_dyn_nmda_dist * (u[3] - E_nmda)


			I_noise += -dt*I_noise/τ_noise + dt**.5*σ_noise*np.random.normal(0.,1.,(n_comp))/τ_noise**.5
			
			I_axial[0] = -g_axial_soma_som_prox_neg*(np.minimum(0.,u[0]-u[1])*n_mult_single_prox + np.minimum(0.,u[0]-u[2])*n_mult_connect_prox)
			
			I_axial[1] = -g_axial_proximal_som_prox_neg*np.minimum(0.,u[1]-u[0]) \
						-g_axial_proximal_som_prox_pos*np.maximum(0.,u[1]-u[0])
			
			I_axial[2] = -g_axial_proximal_som_prox_neg*np.minimum(0.,u[2]-u[0]) \
						-g_axial_proximal_som_prox_pos*np.maximum(0.,u[2]-u[0]) \
						-g_axial_proximal_prox_dist_neg*np.minimum(0.,u[2]-u[3]) \
						-g_axial_proximal_prox_dist_pos*np.maximum(0.,u[2]-u[3])

			I_axial[3] = -g_axial_dist_dist_prox_neg*np.minimum(0.,u[3]-u[2]) \
						-g_axial_dist_dist_prox_pos*np.maximum(0.,u[3]-u[2])
			
			g_dyn_ampa_prox += -dt*g_dyn_ampa_prox/τ_ampa
			g_dyn_nmda_prox += -dt*g_dyn_nmda_prox/τ_nmda

			g_dyn_ampa_dist += -dt*g_dyn_ampa_dist/τ_ampa
			g_dyn_nmda_dist += -dt*g_dyn_nmda_dist/τ_nmda

			spike_prox = 1.*(np.random.rand()<=dt*r_prox)
			spike_dist = 1.*(np.random.rand()<=dt*r_dist)

			g_dyn_ampa_prox += g_ampa*spike_prox*w_prox
			g_dyn_nmda_prox += g_nmda*spike_prox*w_prox

			g_dyn_ampa_dist += g_ampa*spike_dist*w_dist
			g_dyn_nmda_dist += g_nmda*spike_dist*w_dist

			for k in range(n_comp):
				if u[k] < θ and not(spiking_state[k]):
					if k == 1:
						u[k] += dt * (I_L[k] + I_exp[k] + I_axial[k] + I_ext[k] + I_noise[k] + I_ampa_prox + I_nmda_prox)/C
					elif k == 3:
						u[k] += dt * (I_L[k] + I_exp[k] + I_axial[k] + I_ext[k] + I_noise[k] + I_ampa_dist + I_nmda_dist)/C
					else:
						u[k] += dt * (I_L[k] + I_exp[k] + I_axial[k] + I_ext[k] + I_noise[k])/C	
				if u[k] >= θ and not(spiking_state[k]):
					spiking_state[k] = 1.
					if k == 0:
						V_T = V_Tmax
					t_last_spike[k] = t*dt
					u[k] = V_spike
					spike_list[k].append(t*dt)

				if spiking_state[k] and (t*dt - t_last_spike[k]) > Δ_spike:
					spiking_state[k] = 0
					u[k] = V_reset


			V_T += dt * (-(V_T - V_Trest)/τ_VT)


			u_rec[t,:] = u
			V_T_rec[t] = V_T


		output_rate[i,j] = len(spike_list[0])/T

fig_sweep, ax_sweep = plt.subplots(1,1)#
ax_sweep.pcolormesh(output_rate)

t_arr = np.linspace(0.,T,n_t)

fig_u,ax_u = plt.subplots(1,1)
for k in range(n_comp):
	ax_u.plot(t_arr,u_rec[:,k],label=labels[k])
plt.legend()

fig_V_T,ax_V_T = plt.subplots(1,1)
ax_V_T.plot(t_arr,V_T_rec)

fig_spikes,ax_spikes = plt.subplots(1,1)

for k in range(n_comp):
	ax_spikes.plot(np.array(spike_list[k]),np.ones(len(spike_list[k]))*k,'x')

plt.show()

pdb.set_trace()