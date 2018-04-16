#!/usr/bin/env python3

from rnn_prox_dist import *

N_e_range_sweep = [500,1500]
N_e_N_sweep = 2
N_e_list_sweep = list(range(N_e_range_sweep[0],N_e_range_sweep[1],int((N_e_range_sweep[1]-N_e_range_sweep[0])/N_e_N_sweep)))

N_t_range_sweep = [100,500]
N_t_N_sweep = 2
N_t_list_sweep = (10**np.linspace(np.log10(N_t_range_sweep[0]),np.log10(N_t_range_sweep[1]),N_t_N_sweep)).astype("int")

performance_array = np.ndarray((N_e_N_sweep,N_t_N_sweep))

for N_e_it in tqdm(range(N_e_N_sweep)):

	for N_t_it in range(N_t_N_sweep):#tqdm(range(N_t_N_sweep)):

		N_e =  N_e_list_sweep[N_e_it]
		#pdb.set_trace()
		n_t_plast = N_t_list_sweep[N_t_it]
		#pdb.set_trace()

		performance_array[N_e_it,N_t_it] = main_simulation(N_e,int(N_e*.2),n_t_plast)[12][-n_t_analysis:].mean()

#plt.pcolormesh(N_t_list_sweep,N_e_list_sweep,performance_array)
#plt.gca().invert_yaxis()
#plt.colorbar()
#plt.xscale("log")
#plt.show()

#pdb.set_trace()
#np.save("performance_plast.npy",performance_array)