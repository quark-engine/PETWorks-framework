import pandas as pd
import numpy as np
import os

from utils.functions import *


def build_appendix_file(csv_name, num_bucket, domain_threshold, float_threshold, domain_consistency):
	## For missing value, we only deal with numerical value( float/int)
	## For float value(if it is bucketized), it start from bucket-0("" [0, bucketSize] "")
	## For domain_threshold, it couldn't be used for categorical value(string value)
	
	#raw_data = pd.read_csv('data_appendix/' + csv_name + '/' + csv_name +'_consistent.csv', low_memory=False)
	raw_data = pd.read_csv('data_appendix/' + csv_name + '/' + csv_name +'.csv', low_memory=False, float_precision='round_trip')
	record_n, _ = raw_data.shape
	#del raw_data['Time']

	attr_name = raw_data.keys().tolist()
	missing_attr = raw_data.isnull().any()
	specs = {}
	transform2float_collection = {}
	categorical_float = {}
	single_attr = {}
	single_attr['original'] = attr_name
	single_attr['single'] = {}

	for attr in attr_name:
		operation = False
		max_v = 0
		if missing_attr[attr]:
			operation = True

		name_v = raw_data[attr].value_counts().index.tolist()
		max_v = len(name_v)

		if max_v == 1:
			single_attr['single'][attr] = name_v[0]
			del raw_data[attr]
			print(attr, 'is a singel-value attribute')
			continue

		name_v.sort()

		specs[attr] = {}
		transform2float = False
		pass_this = False

		if raw_data[attr].dtypes == 'int64' or raw_data[attr].dtypes == 'int':
			if max_v > domain_threshold:
				transform2float = True

		if raw_data[attr].dtypes == 'float64' or raw_data[attr].dtypes == 'float' or transform2float:
			if domain_consistency:
				if max_v > domain_threshold:
					if operation:
						specs[attr]['optional'] = True
						specs[attr]['max'] = float(raw_data[attr].max())
					else:
						specs[attr]['optional'] = False
						specs[attr]['max'] = float(raw_data[attr].max())
					specs[attr]['min'] = float(raw_data[attr].min())
					specs[attr]['type'] = 'float' 
					pass_this = True
					if transform2float:
						if max_v < float_threshold:
							transform2float_collection[attr] = name_v
						else:
							transform2float_collection[attr] = []
						print(attr, 'is tranformed from int to float with max_value:', specs[attr]['max'])
					elif max_v < float_threshold:
						categorical_float[attr] = name_v
						print(attr, 'is float(categorical) and the max_value is:', specs[attr]['max'])
					else:
						print(attr, 'is float(numerical) and the max_value is:', specs[attr]['max'])
			else:
				if max_v > domain_threshold:
					if operation:
						specs[attr]['optional'] = True
						specs[attr]['max'] = float(raw_data[attr].max())
					else:
						specs[attr]['optional'] = False
						specs[attr]['max'] = float(raw_data[attr].max())
					specs[attr]['min'] = float(raw_data[attr].min())
					specs[attr]['type'] = 'float' 
					pass_this = True
					if transform2float:
						transform2float_collection[attr] = []
						print(attr, 'is tranformed from int to float with max_value:', specs[attr]['max'])
					else:
						print(attr, 'is float(numerical) and the max_value is:', specs[attr]['max'])

		if pass_this:
			continue

		specs[attr]['type'] = 'enum'
		if operation:
			specs[attr]['optional'] = True
		else:
			specs[attr]['optional'] = False
		print(attr, 'with enum')

	mypath = 'data/' + csv_name
	if not os.path.isdir(mypath):
		os.makedirs(mypath)

	store(specs, mypath + '/' + csv_name + '-specs')
	store(single_attr, mypath + '/' + csv_name + '-single_attr')
	store(transform2float_collection, mypath + '/' + csv_name + '-transform2float')
	store(categorical_float, mypath + '/' + csv_name + '-categorical_float')
	raw_data.to_csv(mypath + '/' + csv_name + '.csv', index=0)  #, float_format='%g'
	print('\n[Info] Transformation ----- completed')
	del raw_data, specs, mypath, missing_attr, attr_name