#!/usr/bin/env python3

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.
mpl.rcParams['font.size'] = 22.
mpl.rcParams['figure.autolayout'] = True
#mpl.rcParams['axes.color_cycle'] = ['009BDE', 'FF8800', '00EB8D', 'FBC15E', '8EBA42', 'FFB5B8']
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['patch.facecolor'] = '009BDE'
mpl.rcParams['mathtext.fontset'] = 'stixsans'
#mpl.rcParams['font.sans-serif'] = 'Arial'
#mpl.rcParams['text.latex.preamble'] = ['\\usepackage{upgreek}']
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['mathtext.default'] = 'sf'

default_fig_width = 5.11811

file_format = [".png",".svg",".eps"] ## cycle through file formats and append to string when saving plots

#sim_data_base_folder = "/home/fschubert/work/repos/Elman_Net/plots/"
#sim_data_base_folder = "/home/fschubert/Master/sim_data/"
#sim_data_base_folder = "/media/fschubert/Ohne Titel/sim_data/"

plots_base_folder = "/home/fschubert/work/repos/Elman_Net/plots/" # base folder of plots
#plots_base_folder = "/home/fabian/work/repos/Elman_Net/plots/"
