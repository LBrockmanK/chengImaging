import pandas as pd
import numpy as np
import os
import cnn

def main():
	print("Working")

	# Read Collagen data
	df = dataRead('RamanData\PoC_Data\Collagen',True)
	# Read Fat Data
	df = pd.concat([df, dataRead('RamanData\PoC_Data\Fat',False)], axis=0)

	# Fix index
	df.reset_index(drop=True, inplace=True)

	# Run CNN model test
	cnn.cnnmodel(df)

def dataRead(directory, datatype):
	# Setup dataframe
	df = pd.DataFrame(columns=['Class','Content'])

	# iterate over files in directory
	for filename in os.listdir(directory):
		f = os.path.join(directory, filename)
		# checking if it is a file
		if os.path.isfile(f):
			#load dataframe from csv
			tempFrame = pd.read_csv(f, header=None, names=['Wavelength','Absorption'])
			spectrum = tempFrame.Absorption.values.tolist()
			# Add data to frame
			df.loc[len(df.index)] = [datatype, spectrum] 
	return df

main()