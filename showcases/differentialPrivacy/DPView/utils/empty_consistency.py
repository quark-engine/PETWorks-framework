import numpy as np

from DPView.utils.functions import *
import copy

def empty_consistency(size_marginal_tables, directory_name, temp_root, path_tmp, non_neg_mode):
	marginal_group = load(path_tmp + '/marginal_group')
	attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
	marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
	pi_value_set = load(path_tmp + '/pi_value_set')

	weights = []
	div_weights = []
	for group in marginal_group:
		mul = 1
		for attr in group:
			mul *= (attr_info[attr]['max'] + 1)
		weights.append(mul)
		del mul
	weights = np.array(weights)

	if non_neg_mode == 0:
		div_weights = np.ones(len(weights))
		div_weights = np.divide(div_weights, weights)
	else:	
		for view_idx in range(len(marginal_group)):
			view = marginal_group[view_idx]
			group_name = '/'.join(view)
			div_weights.append(((pi_value_set[group_name])**2)/weights[view_idx])
	div_weights = np.divide(div_weights, np.sum(div_weights))

	sum_count = 0
	each_count = []
	for gp_idx in range(len(marginal_group)):
		view = marginal_group[gp_idx]
		view_name = '/'.join(view)
		split_path = marginal_table_dict[view_name]['savePath']
		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)
		each_count.append(sum(value for _, value in tmp_dict.items()))
		sum_count += each_count[gp_idx] * div_weights[gp_idx]
		del view, view_name, split_path, dict_path, tmp_dict
	each_count = np.array(each_count)
	each_count = sum_count - each_count
	print('each_count:', np.sum(each_count))

	############################[ Evenly ]###############################
	each_count = np.divide(each_count, weights)
	## updating
	for gp_idx in range(len(marginal_group)):
		view = marginal_group[gp_idx]
		view_name = '/'.join(view)
		split_path = marginal_table_dict[view_name]['savePath']
		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		padding_value = each_count[gp_idx]
		tmp_dict = pickle_load(dict_path)
		tmp_dict = {idx:(value+padding_value) for idx, value in tmp_dict.items()}
		pickle_store(tmp_dict, dict_path)
		del view, view_name, split_path, dict_path, tmp_dict, padding_value
	del marginal_group, attr_info, marginal_table_dict, weights, div_weights, each_count
	####################################################################


	############################[ Zero protection ]###############################
	# for gp_idx in range(len(marginal_group)):
	# 	view = marginal_group[gp_idx]
	# 	view_name = '/'.join(view)
	# 	split_path = marginal_table_dict[view_name]['savePath']
	# 	dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
	# 	total_padding_value = copy.deepcopy(each_count[gp_idx])
	# 	tmp_dict = pickle_load(dict_path)
		
	# 	if total_padding_value < 0:
	# 		## padding_value is negative
	# 		while total_padding_value != 0:
	# 			local_tmp_dict = {idx:value for idx, value in tmp_dict.items() if value > 0}
	# 			if len(local_tmp_dict) != 0:
	# 				## find minimal value
	# 				min_value = copy.deepcopy(local_tmp_dict[min(local_tmp_dict, key=local_tmp_dict.get)])
	# 				padding_value = total_padding_value/len(local_tmp_dict)
	# 				if abs(padding_value) <= min_value:
	# 					local_tmp_dict = {idx:(value+padding_value) for idx, value in local_tmp_dict.items()}
	# 					tmp_dict.update(local_tmp_dict)
	# 					total_padding_value = 0
	# 				else:
	# 					local_tmp_dict = {idx:(value-min_value) for idx, value in local_tmp_dict.items()}
	# 					tmp_dict.update(local_tmp_dict)
	# 					total_padding_value += min_value*len(local_tmp_dict)
	# 				del local_tmp_dict, min_value, padding_value
	# 			else:
	# 				padding_value = total_padding_value/weights[gp_idx]
	# 				tmp_dict = {idx:(value+padding_value) for idx, value in tmp_dict.items()}
	# 				total_padding_value = 0
	# 	elif total_padding_value > 0:
	# 		## padding_value is positive
	# 		local_tmp_dict = {idx:value for idx, value in tmp_dict.items() if value > 0}
	# 		if len(local_tmp_dict) != 0:
	# 			padding_value = total_padding_value/len(local_tmp_dict)
	# 			local_tmp_dict = {idx:(value+padding_value) for idx, value in local_tmp_dict.items()}
	# 			tmp_dict.update(local_tmp_dict)
	# 			del local_tmp_dict, padding_value
	# 		else:
	# 			padding_value = total_padding_value/weights[gp_idx]
	# 			tmp_dict = {idx:(value+padding_value) for idx, value in tmp_dict.items()}
	# 	pickle_store(tmp_dict, dict_path)
	# del marginal_group, attr_info, marginal_table_dict, weights, div_weights, each_count
	####################################################################