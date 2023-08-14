import pandas as pd
import numpy as np
import itertools
import copy
import os

from django.utils.translation import gettext
from general.exception import BreakProgramException
from DPView.utils.functions import *
from tqdm import tqdm


def group_common(groupA, groupB):
    common = list(set(groupA).intersection(set(groupB)))
    return common

def group_different(groupA, groupB):
    different = list(set(groupB).difference(set(groupA)))
    return different

def find_neighboring_set(a, domains):
    output = []
    for pos in range(len(a)):
        for val in range(0,a[pos],1):
            output.append(a[:pos]+(val,)+a[pos+1:])
        for val in range(a[pos]+1,domains[pos],1):
            output.append(a[:pos]+(val,)+a[pos+1:])
    return output


def consistency(size_marginal_tables, directory_name, temp_root, path_tmp, non_neg_mode, check_add_intersection_view, file, num_progress_limit):
    intersect_attr_num = size_marginal_tables // 2
    intersect_attr_num_complement = size_marginal_tables % 2
    progress_count = 0
    
    base_intersection_collect = load(path_tmp + '/base_intersection_collect')
    base_views = base_intersection_collect[0]
    intersection_views = base_intersection_collect[1]
    del base_intersection_collect
    
    attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
    marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
    pi_value_set = load(path_tmp + '/pi_value_set')
    
    dict_path2 = path_tmp + '/marginal_tables/group_by_file'
    if not os.path.isdir(dict_path2):
        os.makedirs(dict_path2)
    
    ## build sequential order list
    sequential_work = []
    start_pos = 0
    for int_view in intersection_views:
        sequential_work.append([base_views[start_pos], int_view])
        sequential_work.append([base_views[start_pos+1], int_view])
        start_pos += 1
    if (intersect_attr_num_complement == 1) and (not check_add_intersection_view):
        if len(base_views) > 1:
            sequential_work.append([base_views[-2], base_views[-1]])
    del start_pos, base_views, intersection_views


    ## reverse the list
    sequential_work.reverse()

    
    for work_content in sequential_work:
        common_attrs = group_common(work_content[0], work_content[1])
        common_attrs.sort()
        
        ## compute weight that is contributed by each view in work_content
        weights = []
        div_weights = []
        for view in work_content:
            mul = 1
            for attr in view:
                if attr not in common_attrs:
                    mul *= (attr_info[attr]['max'] + 1)
            weights.append(mul)
            del mul
        weights = np.array(weights)
        if non_neg_mode == 0:
            div_weights = np.ones(len(weights))
            div_weights = np.divide(div_weights, weights)
        else:
            for view_idx in range(len(work_content)):
                view = work_content[view_idx]
                group_name = '/'.join(view)
                div_weights.append(((pi_value_set[group_name])**2)/weights[view_idx])
        div_weights = np.divide(div_weights, np.sum(div_weights))
        
        marginal_table = {}
        attr_sizes = []
        for element in common_attrs:
            attr_sizes.append(range(attr_info[element]['max'] + 1))
        attr_sizes_extend = list(itertools.product(*attr_sizes))
        marginal_table = {ele:0 for ele in attr_sizes_extend}
        del attr_sizes, attr_sizes_extend
        
        for view_idx in range(len(work_content)):
            view = work_content[view_idx]
            group_name = '/'.join(view)
            split_path = marginal_table_dict[group_name]['savePath']
            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)
            
            groupby_pos = [pos for pos in range(len(view)) if view[pos] not in common_attrs]
            groupby_pos.reverse()

            ## build new contingency table
            ##
            new_view = copy.deepcopy(view)
            for sort_pos in groupby_pos:
                progress_count = progress_count + 1
                if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                    progress_count = 0
                    file.num_progress = file.num_progress + 1
                    file.save()
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                ## building new space
                attr_sizes = []
                sort_value = 0
                for pos in range(len(new_view)):
                    if pos != sort_pos:
                        attr_sizes.append(range(attr_info[new_view[pos]]['max'] + 1))
                    else:
                        sort_value = (attr_info[new_view[pos]]['max'] + 1)
                attr_sizes_extend = list(itertools.product(*attr_sizes))
                del attr_sizes

                inner_new_tmp_dict = {ele:0 for ele in attr_sizes_extend}
                for ele in attr_sizes_extend:
                    inner_new_tmp_dict[ele] += sum([tmp_dict[ele[:sort_pos] + (extend,) + ele[sort_pos:]] for extend in range(sort_value)])
                del tmp_dict, attr_sizes_extend
                tmp_dict = copy.deepcopy(inner_new_tmp_dict)
                del inner_new_tmp_dict
                new_view.remove(new_view[sort_pos])
            pickle_store(tmp_dict, dict_path2 + '/' + str(view_idx))
            
            for event in tmp_dict.keys():
                marginal_table[event] += tmp_dict[event] * div_weights[view_idx]
            del group_name, split_path, dict_path, tmp_dict, groupby_pos, new_view, view
        
        ## return result and then update it
        ##
        for view_idx in range(len(work_content)):
            progress_count = progress_count + 1
            if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            view = work_content[view_idx]
            group_name = '/'.join(view)
            split_path = marginal_table_dict[group_name]['savePath']
            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)

            s1 = copy.deepcopy(marginal_table)
            s2 = pickle_load(dict_path2 + '/' + str(view_idx))
            s1 = {idx : s1.get(idx, 0) - s2.get(idx, 0) for idx in set(s1.keys()) | set(s2.keys())}

            del s2, split_path

            ############################[ Evenly ]###############################
            s1 = {idx:(value/weights[view_idx]) for idx, value in s1.items()}

            ## recover attributes
            extend_idx = [pos for pos in range(len(view)) if view[pos] not in common_attrs]
            extend_value = [(attr_info[view[pos]]['max'] + 1) for pos in range(len(view)) if view[pos] not in common_attrs]
            for extend_idx_idx in range(len(extend_idx)):
                s1 = {idx[:extend_idx[extend_idx_idx]] + (extend,) + idx[extend_idx[extend_idx_idx]:]:value for extend in range(extend_value[extend_idx_idx]) for idx, value in s1.items()}
            del extend_idx, extend_value

            tmp_dict = pickle_load(dict_path)
            s2 = {idx:(tmp_dict[idx] + value) for idx, value in s1.items()}

            del s1
            tmp_dict.update(s2)
            del s2

            ## try to update data without non-negativity processs
            pickle_store(tmp_dict, dict_path)
            del group_name, dict_path, view, tmp_dict
        del marginal_table, common_attrs, weights, div_weights
        ####################################################################


            ############################[ Zero protection ]###############################
            # tmp_dict = pickle_load(dict_path)
            # for cell in s1.keys():
            # 	## find neighboring set
            # 	attr_sizes = []
            # 	cell_pos = 0
            # 	for pos in range(len(view)):
            # 		if view[pos] in common_attrs:
            # 			attr_sizes.append([cell[cell_pos]])
            # 			cell_pos += 1
            # 		else:
            # 			attr_sizes.append(range(attr_info[view[pos]]['max'] + 1))
            # 	neighboring = list(itertools.product(*attr_sizes))
            # 	del attr_sizes

            # 	updated_value = s1[cell]
            # 	if updated_value < 0:
            # 		while updated_value != 0:
            # 			local_tmp_dict = {idx:tmp_dict[idx] for idx in neighboring if tmp_dict[idx] > 0}
            # 			if len(local_tmp_dict) != 0:
            # 				## find minimal value
            # 				min_value = copy.deepcopy(local_tmp_dict[min(local_tmp_dict, key=local_tmp_dict.get)])
            # 				padding_value = updated_value/len(local_tmp_dict)
            # 				if abs(padding_value) <= min_value:
            # 					local_tmp_dict = {idx:(value+padding_value) for idx, value in local_tmp_dict.items()}
            # 					tmp_dict.update(local_tmp_dict)
            # 					updated_value = 0
            # 				else:
            # 					local_tmp_dict = {idx:(value-min_value) for idx, value in local_tmp_dict.items()}
            # 					tmp_dict.update(local_tmp_dict)
            # 					updated_value += min_value*len(local_tmp_dict)
            # 				del local_tmp_dict, min_value, padding_value
            # 			else:
            # 				padding_value = updated_value/len(neighboring)
            # 				local_tmp_dict = {idx:(tmp_dict[idx]+padding_value) for idx in neighboring}
            # 				tmp_dict.update(local_tmp_dict)
            # 				updated_value = 0
            # 				del local_tmp_dict
            # 	elif updated_value > 0:
            # 		local_tmp_dict = {idx:tmp_dict[idx] for idx in neighboring if tmp_dict[idx] > 0}
            # 		if len(local_tmp_dict) != 0:
            # 			padding_value = updated_value/len(local_tmp_dict)
            # 			local_tmp_dict = {idx:(value+padding_value) for idx, value in local_tmp_dict.items()}
            # 			tmp_dict.update(local_tmp_dict)
            # 			del local_tmp_dict, padding_value
            # 		else:
            # 			padding_value = updated_value/len(neighboring)
            # 			local_tmp_dict = {idx:(tmp_dict[idx]+padding_value) for idx in neighboring}
            # 			tmp_dict.update(local_tmp_dict)
            # 			del local_tmp_dict
            ####################################################################


            #pickle_store(tmp_dict, dict_path)
            
            ## Do non-negativity process
            ## 221 - 297
            
        # 	if non_neg_mode == 0:
        # 		minValue = tmp_dict[min(tmp_dict, key=tmp_dict.get)]
        # 		tmp_dict = {key: tmp_dict[key] - minValue for key in tmp_dict.keys()}
        # 		pickle_store(tmp_dict, dict_path)
        # 	elif non_neg_mode == 1:
        # 		sub_num = abs(sum(value for _, value in tmp_dict.items() if value < 0))
        # 		total_num = sum(value for _, value in tmp_dict.items() if value > 0)

        # 		if total_num < sub_num:
        # 			tmp_dict = dict.fromkeys(tmp_dict, 0)
        # 		elif sub_num > 0:
        # 			tmp_dict_neg = {idx:0 for idx,value in tmp_dict.items() if value < 0}
        # 			tmp_dict.update(tmp_dict_neg)
        # 			del tmp_dict_neg
        # 			tmp_dict_pos = {idx:value for idx,value in tmp_dict.items() if value > 0}
        # 			tmp_dict_pos = sorted(tmp_dict_pos.items(), key=lambda d: d[1])
        # 			tmp_dict_pos_value = np.array([ele[1] for ele in tmp_dict_pos])
        # 			tmp_dict_pos_value = np.cumsum(tmp_dict_pos_value)
        # 			tmp_dict_pos_zero = np.where(tmp_dict_pos_value < sub_num)[0]
        # 			if len(tmp_dict_pos_zero) > 0:
        # 				tmp_dict_pos_zero = tmp_dict_pos_zero[-1]
        # 			else:
        # 				tmp_dict_pos_zero = -1
        # 				#print(tmp_dict_pos_value[0], ' with sub: ', sub_num)

        # 			remain_value = tmp_dict_pos_value[tmp_dict_pos_zero + 1] - sub_num
        # 			tmp_dict.update({tmp_dict_pos[tmp_dict_pos_zero + 1][0]:remain_value})
        # 			del remain_value
        # 			tmp_dict_zero = {ele[0]:0 for ele in tmp_dict_pos[:tmp_dict_pos_zero + 1]}
        # 			tmp_dict.update(tmp_dict_zero)
        # 			del tmp_dict_zero, tmp_dict_pos, tmp_dict_pos_zero
        # 		pickle_store(tmp_dict, dict_path)
        # 	elif non_neg_mode == 2:
        # 		tmp_dict_zero = {idx:0 for idx,value in tmp_dict.items() if value < 0}
        # 		tmp_dict.update(tmp_dict_zero)
        # 		pickle_store(tmp_dict, dict_path)
        # 		del tmp_dict_zero
        # 	elif non_neg_mode == 3:
        # 		sub_num = abs(sum(value for _, value in tmp_dict.items() if value < 0))
        # 		total_num = sum(value for _, value in tmp_dict.items() if value > 0)

        # 		if total_num < sub_num:
        # 			tmp_dict = dict.fromkeys(tmp_dict, 0)
        # 		elif total_num > sub_num:
        # 			divisor = 0
        # 			domains = []
        # 			for attr in view:
        # 				divisor += attr_info[attr]['max']
        # 				domains.append(attr_info[attr]['max'] + 1)

        # 			neg_cells = [cell for cell, value in tmp_dict.items() if value < 0]

        # 			while len(neg_cells) > 0:
        # 				for cell in neg_cells:
        # 					## list all neighboring cells
        # 					nerghbors = find_neighboring_set(cell, domains)
        # 					local_sub_num = tmp_dict[cell]/divisor
        # 					for neighbor in nerghbors:
        # 						tmp_dict[neighbor] += local_sub_num
        # 					tmp_dict[cell] = 0
        # 					del local_sub_num, nerghbors
        # 				## filtering small count
        # 				neg_cells = [cell for cell, value in tmp_dict.items() if value < 0]
        # 				tmp_dict_zero = {cell:0 for cell in neg_cells if tmp_dict[cell] > -1.e-16}
        # 				if len(tmp_dict_zero) > 0:
        # 					tmp_dict.update(tmp_dict_zero)
        # 					for cell in list(tmp_dict_zero.keys()):
        # 						neg_cells.remove(cell)
        # 				del tmp_dict_zero
        # 			del divisor, domains
        # 		pickle_store(tmp_dict, dict_path)
        # 		del sub_num, total_num
        # 	else:
        # 		pickle_store(tmp_dict, dict_path)
        # 		donothing = True
        # 	del group_name, dict_path, view, tmp_dict
        # del marginal_table, common_attrs, weights, div_weights
        

        for view_idx in range(2):
            os.remove(dict_path2 + '/' + str(view_idx) + '.pickle')
    del sequential_work, dict_path2, attr_info, pi_value_set