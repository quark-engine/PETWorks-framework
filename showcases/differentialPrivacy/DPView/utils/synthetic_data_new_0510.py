import pandas as pd
import numpy as np
import itertools
import operator
import random
import copy
import time
import os

from django.utils.translation import gettext
from general.exception import BreakProgramException
from numpy.random import choice
from DPView.utils.functions import *
from tqdm import tqdm


def group_common(groupA, groupB):
    common = list(set(groupA).intersection(set(groupB)))
    return common
    
def group_different(groupA, groupB):
    different = list(set(groupB).difference(set(groupA)))
    return different

def find_single_view_cases(base_view):
    L1 = itertools.chain(*(itertools.combinations(base_view, n) for n in range(1, len(base_view)+1)))
    L1 = list(L1)
    for case_idx in range(len(L1)):
        L1[case_idx] = list(L1[case_idx])
        L1[case_idx].sort()
    return L1

def find_cases(pre_attrs, post_attrs):
    all_cases = []
    L1 = itertools.chain(*(itertools.combinations(pre_attrs, n) for n in range(1, len(pre_attrs)+1)))
    #L2 = itertools.chain(*(itertools.combinations(post_attrs, n) for n in range(1, len(post_attrs)+1)))
    L2 = itertools.chain(*(itertools.combinations(post_attrs, n) for n in range(1, 2)))
    all_cases = [list(a+b) for a, b in itertools.product(L1, L2)]
    all_cases.append(pre_attrs + post_attrs)
    for case_idx in range(len(all_cases)):
        all_cases[case_idx].sort()
    del L1, L2
    return all_cases

def sort_by_number_and_domain(view, attr_info):
    new_view = []
    view_size = [len(view[pos]) for pos in range(len(view))]
    view_domain = [0]*len(view)
    for idx in range(len(view)):
        mul = 1
        for attr in view[idx]:
            mul *= (attr_info[attr]['max'] + 1)
        view_domain[idx] = mul
        del mul
    
    while len(new_view) != len(view):
        collect_largest_size = [pos for pos in range(len(view_size)) if view_size[pos] == max(view_size)]
        tmp_domain = [view_domain[pos] for pos in collect_largest_size]
        collect_lowest_domain = [pos for pos in collect_largest_size if view_domain[pos] == min(tmp_domain)]
        pick_pos = random.sample(collect_lowest_domain, 1)[0]
        new_view.append(view[pick_pos])
        view_size[pick_pos] = -1
        view_domain[pick_pos] = 99999999
        del collect_largest_size, tmp_domain, collect_lowest_domain, pick_pos
    del view_size, view_domain
    return new_view

def synthesize_data(size_marginal_tables, num_marginal_tables, record_n, directory_name, temp_root, path_tmp, attr_name_list, check_add_intersection_view, file, num_progress_limit):
    target = record_n
    progress_count = 0
    
    check_case = 0  # complete splitting
    if len(attr_name_list) % size_marginal_tables != 0:
        check_case = 1  # incomplete splitting

    marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
    attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
    marginal_groups = load(path_tmp + '/marginal_group2')
    base_intersection_collect = load(path_tmp + '/base_intersection_collect')
    base_groups = base_intersection_collect[0]
    intersection_groups = base_intersection_collect[1]
    del base_intersection_collect
    
    
    ## build initial synthetic data (the first block)
    ##
    synthetic_data = []
    group_name = '/'.join(base_groups[0])
    split_path = marginal_table_dict[group_name]['savePath']
    dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
    tmp_dict = pickle_load(dict_path)
    tmp_dict = {idx:value for idx,value in tmp_dict.items() if value > 0}

    syn_idx = 0
    total_time = time.time()

    tmp_dict_records = list(tmp_dict.keys())
    for gen_r in range(target):
        pick = random.sample(tmp_dict_records, 1)[0]
        synthetic_data.append({})
        for ele in range(size_marginal_tables):
            synthetic_data[syn_idx][base_groups[0][ele]] = pick[ele]
        if tmp_dict[pick] <= 1:
            del tmp_dict[pick]
            tmp_dict_records.remove(pick)
        else:
            tmp_dict[pick] = tmp_dict[pick] - 1
        del pick
        syn_idx += 1
    del tmp_dict, group_name, split_path, dict_path, tmp_dict_records, syn_idx
    remaining_attrs = copy.deepcopy(attr_name_list)
    remaining_attrs = [attr for attr in remaining_attrs if attr not in base_groups[0]]
    
    ## combine with other blocks
    ##
    collect_views = []
    for view_pos in range(len(base_groups)):
        collect_views.append(base_groups[view_pos])
        if view_pos < len(intersection_groups):
            collect_views.append(intersection_groups[view_pos])
    
    pivot = copy.deepcopy(collect_views[0])
    print('[INFO] Starting combine with other blocks...')
    #for insert_view in tqdm(collect_views[1:], ncols = 90):
    for insert_view in collect_views[1:]:        
        group_name = '/'.join(insert_view)
        split_path = marginal_table_dict[group_name]['savePath']
        dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
        bs_dict = pickle_load(dict_path)
        bs_dict = {idx:value for idx,value in bs_dict.items() if value > 0}
        bs_dict_cells = list(bs_dict.keys())
        
        condition_attrs = group_common(pivot, insert_view)
        learned_attrs = group_different(pivot, insert_view)
        all_cases = find_single_view_cases(condition_attrs)
        all_cases = sort_by_number_and_domain(all_cases, attr_info)
        remaining = list(range(target))
        print('='*90)
        print('attributes:', insert_view)
        for case in all_cases:
            print(case, 'with remaining -', len(remaining))
            if len(remaining) == 0:
                break
            new_remaining = []
            for rd_idx in remaining:
                attr_sizes = []
                for attr in insert_view:
                    if attr in case:
                        attr_sizes.append([synthetic_data[rd_idx][attr]])
                    else:
                        attr_sizes.append(range(attr_info[attr]['max'] + 1))
                attr_sizes_extend = list(itertools.product(*attr_sizes))
                del attr_sizes
                
                attr_sizes_extend2 = [att for att in attr_sizes_extend if att in bs_dict_cells]
                if len(attr_sizes_extend2) != 0:
                    count_choices = [bs_dict[cells] for cells in attr_sizes_extend2]
                    pick_idx = [cells_idx for cells_idx in range(len(count_choices)) if count_choices[cells_idx] == max(count_choices)]
                    pick_idx = random.sample(pick_idx, 1)[0]
                    
                    start_pos_idx = 0
                    for attr in insert_view:
                        progress_count = progress_count + 1
                        if progress_count%30000 == 0 and file.num_progress < num_progress_limit:
                            progress_count = 0
                            file.num_progress = file.num_progress + 1
                            file.save()
                        if file.skip:
                            raise BreakProgramException(gettext('程式成功終止'))
                        if attr in learned_attrs:
                            synthetic_data[rd_idx][insert_view[start_pos_idx]] = attr_sizes_extend2[pick_idx][start_pos_idx]
                        start_pos_idx += 1		
                    ## update it
                    if count_choices[pick_idx] <= 1:
                        del bs_dict[attr_sizes_extend2[pick_idx]]
                        bs_dict_cells.remove(attr_sizes_extend2[pick_idx])
                    else:
                        bs_dict[attr_sizes_extend2[pick_idx]] -= 1
                    del count_choices, pick_idx, start_pos_idx
                else:
                    new_remaining.append(rd_idx)
                del attr_sizes_extend, attr_sizes_extend2
            remaining = copy.deepcopy(new_remaining)
            del new_remaining
        print('random pick with remaining -', len(remaining))
        if len(remaining) != 0:
            ## random pick
            remain_count = len(remaining)
            for rd_idx in remaining:
                ## build probabilistic list
                count_choices = np.array([bs_dict[cells] for cells in bs_dict_cells])
                count_choices = np.divide(count_choices, remain_count)
                pick_idx = np.random.choice(range(len(bs_dict_cells)), p = count_choices)
                
                start_pos_idx = 0
                for attr in insert_view:
                    progress_count = progress_count + 1
                    if progress_count%30000 == 0 and file.num_progress < num_progress_limit:
                        progress_count = 0
                        file.num_progress = file.num_progress + 1
                        file.save()
                    if file.skip:
                        raise BreakProgramException(gettext('程式成功終止'))
                    if attr in remaining_attrs:
                        synthetic_data[rd_idx][insert_view[start_pos_idx]] = bs_dict_cells[pick_idx][start_pos_idx]
                    start_pos_idx += 1					
                if bs_dict[bs_dict_cells[pick_idx]] <= 1:
                    del bs_dict[bs_dict_cells[pick_idx]]
                    bs_dict_cells.remove(bs_dict_cells[pick_idx])
                else:
                    bs_dict[bs_dict_cells[pick_idx]] -= 1
                remain_count -= 1
                del count_choices, pick_idx
        del remaining
        pivot = pivot + insert_view
    print('=' * 90)
    print('Total time:', time.time() - total_time)
    synthetic_data = pd.DataFrame(synthetic_data, columns = attr_name_list)
    synthetic_data.to_csv(path_tmp + '/' + directory_name + '_synthetic.csv', index=0)
    synthetic_length = len(synthetic_data)
    del attr_info, synthetic_data, target
    print('synthetic_length:', synthetic_length)
    
def synthesize_data_flat(directory_name, temp_root, path_tmp, attr_name, record_n, random_gen, file, num_progress_limit):
    target = record_n
    progress_count = 0

    synthetic_data = []
    if not random_gen:
        dict_path = path_tmp + '/marginal_tables/marginal_table'
        tmp_dict = pickle_load(dict_path)
        tmp_dict = {idx:value for idx,value in tmp_dict.items() if value > 0}

        syn_idx = 0
        tmp_dict_records = list(tmp_dict.keys())
        for gen_r in range(target):
            progress_count = progress_count + 1
            if progress_count%30000 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            pick = random.sample(tmp_dict_records, 1)[0]
            synthetic_data.append({})
            for ele in range(len(attr_name)):
                synthetic_data[syn_idx][attr_name[ele]] = pick[ele]
            if tmp_dict[pick] <= 1:
                del tmp_dict[pick]
                tmp_dict_records.remove(pick)
            else:
                tmp_dict[pick] = tmp_dict[pick] - 1
            del pick
            syn_idx += 1
        del tmp_dict, dict_path, tmp_dict_records, syn_idx
    else:
        attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
        candidates = list(range(attr_info[attr_name[0]]))
        for gen_r in range(target):
            progress_count = progress_count + 1
            if progress_count%30000 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            synthetic_data.append({})
            synthetic_data[gen_r][attr_name[0]] = random.sample((range(attr_info[attr_name[0]]['max']+1), 1))[0]
        for attr in attr_name[1:]:
            for gen_r in range(target):
                synthetic_data[gen_r][attr] = random.sample((range(attr_info[attr]['max']+1), 1))[0]
        del attr_info
    synthetic_data = pd.DataFrame(synthetic_data, columns = attr_name)
    synthetic_data.to_csv(path_tmp + '/' + directory_name + '_synthetic.csv', index=0)
    synthetic_length = len(synthetic_data)
    synthetic_data, target
    print('synthetic_length:', synthetic_length)