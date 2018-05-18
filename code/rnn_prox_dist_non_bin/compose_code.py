#!/usr/bin/env python3

import csv
import sys
import ast

param_dict = {}


params_path = "./sim_parameters/"+sys.argv[1]+".csv"

param_dict = {}

with open(params_path,"r") as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		param_dict[row[0]] = ast.literal_eval(row[1])


staging = param_dict["staging"]

codestr = []

flag_count = 0

with open("rnn_prox_dist_non_bin.py","r") as codefile:
	for row in codefile:
		flag_count += 1
		codestr.append(row)
		if '###CUSTOM_COMMANDS_FLAG###' in row:
			flag_line = flag_count
		

#import pdb
#pdb.set_trace()

#codestr_comp = ''''''



for k in range(len(staging[0])):
	
	if_stat = ""
	for l in range(codestr[flag_line-1].count('\t')):
		if_stat += '\t'
	if_stat += "if t==" + str(staging[0][k]) + ":\n"
	
	cmd = ""
	for l in range(codestr[flag_line-1].count('\t') + 1):
		cmd += '\t'
	cmd += str(staging[1][k]) + "\n"
	
	codestr.insert(flag_line,cmd)
	codestr.insert(flag_line,if_stat)

	#codestr_comp

#for line in codestr:
#	codestr_comp += line

#print(flag_line)

with open("code_temp.py","w") as temp_codefile:
	for line in codestr:
		temp_codefile.write(line)


#print(codestr_comp)