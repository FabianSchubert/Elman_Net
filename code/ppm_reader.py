#!/usr/bin/env python3
import numpy as np
import csv
import pdb

def read_ppm(file_name):

	data = []

	with open(file_name,"rb") as datafile:
		csv_reader = csv.reader(datafile,delimiter=" ")
		next(csv_reader)
		next(csv_reader)

		for row in csv_reader:
			data.append(row)

		res = np.array(data[0]).astype("int")

		data_vec = np.array(data[2:]).astype("float")
		data_arr = np.reshape(data_vec[::3],res)
		data_arr = data_arr/data_arr.max()
		
		return data_arr

