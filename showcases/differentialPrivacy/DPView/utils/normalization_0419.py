from DPView.utils.functions import *

import copy
import os


def delete_zero_sum(marginal_group, zero_sum):
    copy_marginal_group = copy.deepcopy(marginal_group)
    for group in marginal_group:
        if group in zero_sum:
            copy_marginal_group.remove(group)
    return copy_marginal_group

def normalization_tmp1(path_tmp, record_n):
	marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
	marginal_group = load(path_tmp + '/marginal_group')

	for group in marginal_group:
		group_name = '/'.join(group)
		split_path = marginal_table_dict[group_name]['savePath']

		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)

		total_num = sum(value for _, value in tmp_dict.items())
		if total_num != 0:
			total_num = total_num/record_n

			tmp_dict = {idx:(value/total_num) for idx, value in tmp_dict.items()}
			tmp_dict_int = {idx:int(value) for idx,value in tmp_dict.items()}
			tmp_dict_point = {idx:(tmp_dict[idx] - value) for idx,value in tmp_dict_int.items()}
			total_num = sum(value for _, value in tmp_dict_int.items())
			padding = record_n - total_num

			for indx in range(padding):
				max_idx = max(tmp_dict_point, key = tmp_dict_point.get)
				tmp_dict_int[max_idx] += 1
				tmp_dict_point[max_idx] = 0
			pickle_store(tmp_dict_int, dict_path)
			del tmp_dict_int, tmp_dict_point, padding
		#print(group, sum(value for _, value in tmp_dict.items()))
		del tmp_dict, group_name, split_path, dict_path, total_num
	del marginal_table_dict

def normalization(path_tmp, attr_name):
	marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
	marginal_group = load(path_tmp + '/marginal_group')

	zero_sum = []
	for group in marginal_group:
		group_name = '/'.join(group)
		split_path = marginal_table_dict[group_name]['savePath']

		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)

		total_num = sum(value for _, value in tmp_dict.items())
		if total_num == 0:
			zero_sum.append(group)
		print(group, sum(value for _, value in tmp_dict.items()))
		del tmp_dict
	del marginal_table_dict
	uncover_attrs = copy.deepcopy(attr_name)
	output = delete_zero_sum(marginal_group, zero_sum)
	outputs = {}
	outputs['output'] = output
	outputs['zero_sum'] = zero_sum
	for group in output:
		for attr in group:
			uncover_attrs = [x for x in uncover_attrs if x != attr]
	if len(uncover_attrs) != 0:
		print('\n\n[Info] Marginal groups cannot cover all attributes.\n')
		outputs['uncover'] = 1
		outputs['uncover_attr'] = uncover_attrs
	else:
		outputs['uncover'] = 0
		outputs['uncover_attr'] = []
	store(outputs, path_tmp + '/marginal_group2')
	del output, outputs, uncover_attrs

def normalization_flat(path_tmp, record_n, group):
	dict_path = path_tmp + '/marginal_tables/marginal_table'
	tmp_dict = pickle_load(dict_path)

	total_num = sum(value for _, value in tmp_dict.items())
	if total_num > 0:
		total_num = total_num/record_n

		tmp_dict = {idx:(value/total_num) for idx, value in tmp_dict.items()}
		tmp_dict_int = {idx:int(value) for idx,value in tmp_dict.items()}
		tmp_dict_point = {idx:(tmp_dict[idx] - value) for idx,value in tmp_dict_int.items()}
		total_num = sum(value for _, value in tmp_dict_int.items())
		padding = record_n - total_num

		for indx in range(padding):
			max_idx = max(tmp_dict_point, key = tmp_dict_point.get)
			tmp_dict_int[max_idx] += 1
			tmp_dict_point[max_idx] = 0
		pickle_store(tmp_dict_int, dict_path)
		del tmp_dict_int, tmp_dict_point, padding
		del tmp_dict, dict_path, total_num
		return False
	else:		
		del tmp_dict, dict_path, total_num
		return True