import pandas as pd
import os

from DPView.utils.functions import *
from tqdm import tqdm


def empirical_counting(directory_name, temp_root, path_tmp):
	raw_data = pd.read_csv(temp_root + directory_name + '/' + directory_name + '_transform.csv', low_memory=False)
	marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
	marginal_group = load(path_tmp + '/marginal_group')
	marginal_group_subValue = {}
	
	for group in tqdm(marginal_group, ncols = 90):
		group_name = '/'.join(group)
		print(group)
		marginal_count = raw_data.groupby(group)[group[0]].count()
		marginal_count = marginal_count.to_dict()
		split_path = marginal_table_dict[group_name]['savePath']
		marginal_group_subValue[group_name] = {}

		dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
		tmp_dict = pickle_load(dict_path)

		for event in marginal_count.keys():
			tmp_dict[event] = tmp_dict[event] + marginal_count[event]
		pickle_store(tmp_dict, dict_path)
		marginal_group_subValue[group_name]['minValue'] = tmp_dict[min(tmp_dict, key=tmp_dict.get)]
		marginal_group_subValue[group_name]['negSum'] = abs(sum(value for _, value in tmp_dict.items() if value < 0))
		marginal_group_subValue[group_name]['posSum'] = sum(value for _, value in tmp_dict.items() if value > 0)
		pickle_store(marginal_group_subValue, path_tmp + '/marginal_group_sub_num')
		
	del raw_data, marginal_table_dict, marginal_group_subValue, tmp_dict, marginal_group

def empirical_counting_flat_vector(directory_name, temp_root, path_tmp, group):
    raw_data = pd.read_csv(temp_root + directory_name + '/' + directory_name + '_transform.csv', low_memory=False)
    
    marginal_count = raw_data.groupby(group)[group[0]].count()
    marginal_count = marginal_count.to_dict()

    dict_path = path_tmp + '/marginal_tables/marginal_table'
    tmp_dict = pickle_load(dict_path)

    for event in marginal_count.keys():
        tmp_dict[event] = tmp_dict[event] + marginal_count[event]
    pickle_store(tmp_dict, dict_path)
        
    del raw_data, tmp_dict, marginal_count