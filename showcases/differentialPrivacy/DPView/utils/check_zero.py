from DPView.utils.functions import *


def check_values(num_marginal_tables, path_tmp, record_n):
	marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
	base_intersection_collect = load(path_tmp + '/base_intersection_collect')
	base_views = base_intersection_collect[0]
	intersection_views = base_intersection_collect[1]
	del base_intersection_collect

	base_count = 0
	intersect_count = 0
	for group in base_views:
		group_name = '/'.join(group)
		split_path = marginal_table_dict[group_name]['savePath']
		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)
		sum_num = sum(value for _, value in tmp_dict.items())
		print(group, ':', sum_num)
		base_count += abs(sum_num - record_n)
		del sum_num, tmp_dict
	for group in intersection_views:
		group_name = '/'.join(group)
		split_path = marginal_table_dict[group_name]['savePath']
		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)
		sum_num = sum(value for _, value in tmp_dict.items())
		print(group, ':', sum_num)
		intersect_count += abs(sum_num - record_n)
		del sum_num, tmp_dict
	
	print('-' * 90)
	print('base distance:', base_count)
	print('intersect distance:', intersect_count)
	print('total distance:', base_count + intersect_count)
	del marginal_table_dict, base_count, intersect_count

def check_values_return(path_tmp, record_n):
	marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
	marginal_group = load(path_tmp + '/marginal_group')

	acc_distance = 0
	for group in marginal_group:
		group_name = '/'.join(group)
		split_path = marginal_table_dict[group_name]['savePath']
		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)
		sum_num = sum(value for _, value in tmp_dict.items())
		acc_distance += abs(sum_num - record_n)
		del sum_num, tmp_dict
	del marginal_table_dict, marginal_group
	return acc_distance