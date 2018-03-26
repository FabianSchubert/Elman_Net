#!/usr/bin/env python3
import numpy as np
import csv
import pdb

def read_ppm(file_name):

	data = []

	with open(file_name,"r") as datafile:
		csv_reader = csv.reader(datafile,delimiter=" ")
		#pdb.set_trace()
		# skip the first two rows in the file (Only Metadata)
		next(csv_reader)
		next(csv_reader)

		for row in csv_reader:
			# append row data to list
			data.append(row)

		# first line (third in the file) contains the "image" size in the order <number of columns>,<number of rows> (should be equal for our purpose anyway)
		res = np.array(data[0]).astype("int")

		# The rest of the actual image data is stored such that each row contains one number.
		# Three subsequent rows make up one triplet of color information.
		# Triplets of color were flattened by going through the image row by row.
		data_vec = np.array(data[2:]).astype("float")
		# Reshape the vector into a 2d array, again row by row, which is the standard for numpy.reshape
		data_arr = np.reshape(data_vec[::3],(res[1],res[0]))

		# check whether the columns are normalizable to sum up to 1 (which is necessary for being interpreted as a transition matrix)
		if not(0. in data_arr.sum(axis=0)):
			data_arr = data_arr/data_arr.sum(axis=0)
			return data_arr
		else:
			raise SystemExit("Error in transition matrix: Column(s) with zero sum found, can not be normalized.")
		

def read_letter_map(file_name):

	data = []

	with open(file_name,"r") as datafile:

		csv_reader = csv.reader(datafile,delimiter=",")

		for row in csv_reader:
			if row[0][0] == "#":
				pass
			else:
				data.append(row)

		node_ind = np.array(data[0]).astype("int")
		letters = np.array(data[1]).astype("str")
		inp_node_ind = np.array(data[2]).astype("int")

		return node_ind, letters, inp_node_ind