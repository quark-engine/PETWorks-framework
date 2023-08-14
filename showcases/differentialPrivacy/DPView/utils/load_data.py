import pandas as pd
import numpy as np
import os

from utils.functions import *


def preprocess(csv_name, specs_name, num_bucket, privacy_budget, allocation_mode):
	
	mypath = 'data/' + csv_name + '/tmp'
	if not os.path.isdir(mypath):
		os.makedirs(mypath)
	del mypath

	raw_data = pd.read_csv('data/' + csv_name + '/' + csv_name + '.csv', low_memory=False, float_precision='round_trip')
	attr_name = raw_data.keys().tolist()
	attr_info = load('data/' + csv_name + '/'  + specs_name)
	attr_info_max = {}
	specs_mapping = {}
	
	for attr in attr_name:
		attr_info_max[attr] = {}
		specs_mapping[attr] = {}
		if attr_info[attr]['type'] != 'enum':
			bucket_size = (attr_info[attr]['max'] - attr_info[attr]['min']) / num_bucket  ## + 0.00001)
			if attr_info[attr]['optional']:
				max_v = attr_info[attr]['max'] + bucket_size
				raw_data[attr].fillna(max_v, inplace = True)
				attr_info_max[attr]['max'] = num_bucket
			else:
				attr_info_max[attr]['max'] = num_bucket - 1
			raw_data[attr] = raw_data[attr] - attr_info[attr]['min']
			raw_data[attr] = raw_data[attr] / bucket_size
			raw_data[attr] = raw_data[attr].apply(np.ceil)
			raw_data[attr].replace(0, 1, inplace = True)  # add 0 to (0,1]
			raw_data[attr] = raw_data[attr] - 1           # count bucket from 0 to (num_bucket-1) + 1
			raw_data[attr].replace(attr_info_max[attr]['max'] + 1, attr_info_max[attr]['max'], inplace = True)  ## address the float problem in pandas
			raw_data[attr] = raw_data[attr].astype('int')
			print(attr, '(float) with max_value:', attr_info_max[attr]['max'])
			del bucket_size
		else:
			raw_data[attr] = raw_data[attr].astype('str')
			name_v = raw_data[attr].value_counts().index.tolist()
			max_v = len(name_v)
			name_v.sort()

			specs_mapping[attr]['count'] =  max_v
			specs_mapping[attr]['dict'] =  {}
			for idx in range(max_v):
				raw_data[attr].replace(name_v[idx], idx, inplace = True)
				specs_mapping[attr]['dict'][str(idx)] = name_v[idx]
			if attr_info[attr]['optional']:
				raw_data[attr].fillna(max_v, inplace = True)
				specs_mapping[attr]['dict'][str(max_v)] = np.nan
				max_v += 1

			attr_info_max[attr]['max'] = max_v - 1
			raw_data[attr] = raw_data[attr].astype('int')
			del name_v, max_v
			print(attr, '(enum) with max_value:', attr_info_max[attr]['max'])

	store(attr_info_max, 'data/' + csv_name + '/' + csv_name +'_attr_info_max')
	store(specs_mapping, 'data/' + csv_name + '/' + csv_name + '-specs-mapping')
	raw_data.to_csv('data/' + csv_name + '/' + csv_name + '_transform.csv', index=0)
	print()
	print('[Info] Save ----- completed')
	del raw_data, attr_name, attr_info_max, specs_mapping

def recover_data(csv_name, specs_name, specs_mapping_name, transform2float_name, categorical_float_name, single_attr_name, path_tmp, allocation_mode, non_neg_mode, privacy_budget_name, size_marginal_tables, num_marginal_tables, num_bucket, synthetic_num):
	raw_data = pd.read_csv(path_tmp + '/' + csv_name + '_synthetic.csv', low_memory=False)
	attr_name = raw_data.keys().tolist()
	dictionary = load('data/' + csv_name + '/'  + specs_mapping_name)
	attr_info = load('data/' + csv_name + '/'  + specs_name)
	single_attrs = load('data/' + csv_name + '/'  + single_attr_name)

	for attr in attr_name:
		if attr_info[attr]['type'] != 'enum':
			bucket_size = (attr_info[attr]['max'] - attr_info[attr]['min']) / num_bucket  ## + 0.00001)
			missin_exist = False
			if attr_info[attr]['optional']:
				if num_bucket in raw_data[attr]:
					null_list = list(raw_data.loc[raw_data[attr] == num_bucket, attr].index)
					missin_exist = True
			raw_data[attr] = raw_data[attr] * bucket_size
			raw_data[attr] = raw_data[attr] + attr_info[attr]['min']
			noise = np.random.uniform(0, bucket_size, raw_data.shape[0])
			raw_data[attr] = raw_data[attr] + noise
			raw_data[attr] = raw_data[attr].astype('float')
			if attr_info[attr]['optional'] and missin_exist:
				raw_data.loc[null_list, attr] = np.nan
				del null_list
			del noise
		else:
			for idx in range(dictionary[attr]['count']):
				if idx in raw_data[attr]:
					raw_data.loc[raw_data[attr] == idx, attr] = dictionary[attr]['dict'][str(idx)]
	single_attr = list(single_attrs['single'].keys())
	if len(single_attr) != 0:
		for attr in single_attr:
			raw_data[attr] = single_attrs['single'][attr]
		raw_data = raw_data.reindex(columns = single_attrs['original'])
	print(raw_data.head())

	transform2float = load('data/' + csv_name + '/'  + transform2float_name)
	attrs = list(transform2float.keys())
	print(attrs)
	if len(attrs) != 0:
		for attr_ in attrs:
			if transform2float[attr_] != []:
				candidates = np.array(transform2float[attr_])
				for idx in range(synthetic_num):
					tmp_value = raw_data.loc[idx, attr_]
					if tmp_value not in candidates:
						raw_data.loc[idx, attr_] = candidates[(np.abs(candidates - tmp_value)).argmin()]
					del tmp_value
				del candidates
			else:
				raw_data[attr_] = raw_data[attr_].astype('int')
	del transform2float

	categorical_float = load('data/' + csv_name + '/'  + categorical_float_name)
	attrs = list(categorical_float.keys())
	if len(attrs) != 0:
		for attr_ in attrs:
			candidates = np.array(categorical_float[attr_])
			for idx in range(synthetic_num):
				tmp_value = raw_data.loc[idx, attr_]
				if tmp_value not in candidates:
					raw_data.loc[idx, attr_] = candidates[(np.abs(candidates - tmp_value)).argmin()]
				del tmp_value
			del candidates
	del categorical_float

	original_data = pd.read_csv('data_appendix/' + csv_name + '/' + csv_name +'.csv', low_memory=False, float_precision='round_trip')
	for attr in single_attrs['original']:
		raw_data[attr] = raw_data[attr].astype(original_data[attr].dtypes)

	raw_data.to_csv('data/' + csv_name + '/' + csv_name + '_' + privacy_budget_name + '_am_' + str(allocation_mode) + '_ngm_' + str(non_neg_mode) + '_'  + str(size_marginal_tables) + '_' + str(num_marginal_tables) +'.csv', index=0, float_format='%g')
	print('\n[Info] Data transformation ----- completed')
	del raw_data, attr_name, dictionary, attr_info, single_attrs, original_data