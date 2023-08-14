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
from sklearn.utils import shuffle
from DPView.utils.functions import *
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

def group_common(groupA, groupB):
    common = list(set(groupA).intersection(set(groupB)))
    # different = list(set(groupB).difference(set(groupA)))
    return common
    
def group_different(groupA, groupB):
    #common = list(set(groupA).intersection(set(groupB)))
    different = list(set(groupB).difference(set(groupA)))
    return different

def index_of_cell(tuple_cell, domain):
    tuple_cell_list = list(tuple_cell)
    idx = 0
    mul = [1]
    val = 1
    for pos in range(len(domain)-1, 0, -1):
        val *= domain[pos]
        mul.append(val)
    for pos in range(len(domain)):
        idx = idx + tuple_cell[pos] * mul[(len(domain) - pos - 1)]
    del tuple_cell_list, mul, val
    return idx

def find_cases(next_base_view, intersection_view):
    post_attrs = group_common(next_base_view, intersection_view)
    pre_attrs = group_different(post_attrs, intersection_view)
    
    L1 = itertools.chain(*(itertools.combinations(pre_attrs, n) for n in range(1, len(pre_attrs)+1)))
    #L2 = itertools.chain(*(itertools.combinations(post_attrs, n) for n in range(1, len(post_attrs)+1)))
    #L1 = itertools.chain(*(itertools.combinations(pre_attrs, n) for n in range(1, 2)))
    L2 = itertools.chain(*(itertools.combinations(post_attrs, n) for n in range(1, 2)))

    all_cases = [list(a+b) for a, b in itertools.product(L1, L2)]
    all_cases.append(intersection_view)
    for case_idx in range(len(all_cases)):
        all_cases[case_idx].sort()
    # if len(all_cases) > 1:
    # 	all_cases.remove(all_cases[0])
    del post_attrs, pre_attrs, L1, L2
    return all_cases

def sort_by_domain(view, attr_info):
    domains = [0]*len(view)
    domains_list = []
    for idx in range(len(view)):
        mul = 1
        tmp_list = []
        for attr in view[idx]:
            mul *= (attr_info[attr]['max'] + 1)
            tmp_list.append(attr_info[attr]['max'] + 1)
        domains[idx] = mul
        domains_list.append(tmp_list)
        del mul, tmp_list
    new_view = []
    new_domain_size = []
    new_domain_size_list = []
    for run in range(len(view)):
        pick_idx, pick_size = min(enumerate(domains), key=operator.itemgetter(1))
        new_view.append(view[pick_idx])
        new_domain_size.append(pick_size)
        new_domain_size_list.append(domains_list[pick_idx])
        domains[pick_idx] = 9999999999
        del pick_idx
    del domains, domains_list
    return new_view, new_domain_size, new_domain_size_list

def find_single_view_cases(base_view):
    L1 = itertools.chain(*(itertools.combinations(base_view, n) for n in range(1, len(base_view)+1)))
    L1 = list(L1)
    for case_idx in range(len(L1)):
        L1[case_idx] = list(L1[case_idx])
        L1[case_idx].sort()
    return L1

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

def synthesize_data_random(size_marginal_tables, num_marginal_tables, record_n, directory_name, temp_root, path_tmp, attr_name_list, check_add_intersection_view, file, num_progress_limit):	
    target = record_n
    progress_count = 0
    
    check_case = 0  # complete splitting
    if len(attr_name_list) % size_marginal_tables != 0:
        check_case = 1  # incomplete splitting

    marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
    attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')

    base_intersection_collect = load(path_tmp + '/base_intersection_collect')
    base_groups = base_intersection_collect[0]
    intersection_groups = base_intersection_collect[1]
    del base_intersection_collect
    
    new_base_view = []
    new_intersection_view = []
    
    ## build initial synthetic data
    ##
    synthetic_data = []
    
    group_name = '/'.join(base_groups[0])
    split_path = marginal_table_dict[group_name]['savePath']
    dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
    tmp_dict = pickle_load(dict_path)
    tmp_dict = {idx:value for idx,value in tmp_dict.items() if value > 0}

    syn_idx = 0
    remaining_attrs = copy.deepcopy(attr_name_list)

    tmp_dict_records = list(tmp_dict.keys())
    while len(tmp_dict) != 0:
        progress_count = progress_count + 1
        if progress_count%100 == 0 and file.num_progress < num_progress_limit:
            progress_count = 0
            file.num_progress = file.num_progress + 1
            file.save()
        if file.skip:
            raise BreakProgramException(gettext('程式成功終止'))
        pick = random.sample(tmp_dict_records, 1)[0]
        synthetic_data.append({})
        for ele in range(size_marginal_tables):
            synthetic_data[syn_idx][base_groups[0][ele]] = pick[ele]
        tmp_dict[pick] = tmp_dict[pick] - 1
        if tmp_dict[pick] == 0:
            del tmp_dict[pick]
            tmp_dict_records.remove(pick)
        del pick
        syn_idx += 1
    del tmp_dict, group_name, split_path, dict_path, tmp_dict_records, syn_idx
    remaining_attrs = [attr for attr in remaining_attrs if attr not in base_groups[0]]

    ## combine with other blocks
    ##
    if check_case == 0:
        for group in base_groups[1:]:
            remaining_attrs = [attr for attr in remaining_attrs if attr not in group]
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']
            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)
            tmp_dict = {idx:value for idx,value in tmp_dict.items() if value > 0}
            
            tmp_dict_records = list(tmp_dict.keys())
            syn_idx = 0
            while syn_idx < record_n:
                progress_count = progress_count + 1
                if progress_count%100 == 0 and file.num_progress < num_progress_limit:
                    progress_count = 0
                    file.num_progress = file.num_progress + 1
                    file.save()
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                pick = random.sample(tmp_dict_records, 1)[0]
                for ele in range(size_marginal_tables):
                    synthetic_data[syn_idx][group[ele]] = pick[ele]
                syn_idx += 1
                tmp_dict[pick] = tmp_dict[pick] - 1
                if tmp_dict[pick] == 0:
                    del tmp_dict[pick]
                    tmp_dict_records.remove(pick)
                del pick
            del tmp_dict, group_name, split_path, dict_path, tmp_dict_records, syn_idx
    else:
        for group in base_groups[1:-1]:
            remaining_attrs = [attr for attr in remaining_attrs if attr not in group]
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']
            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)
            tmp_dict = {idx:value for idx,value in tmp_dict.items() if value > 0}
            
            tmp_dict_records = list(tmp_dict.keys())
            syn_idx = 0
            while syn_idx < record_n:
                progress_count = progress_count + 1
                if progress_count%1000 == 0 and file.num_progress < num_progress_limit:
                    progress_count = 0
                    file.num_progress = file.num_progress + 1
                    file.save()
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                pick = random.sample(tmp_dict_records, 1)[0]
                for ele in range(size_marginal_tables):
                    synthetic_data[syn_idx][group[ele]] = pick[ele]
                syn_idx += 1
                tmp_dict[pick] = tmp_dict[pick] - 1
                if tmp_dict[pick] == 0:
                    del tmp_dict[pick]
                    tmp_dict_records.remove(pick)
                del pick
            del tmp_dict, group_name, split_path, dict_path, tmp_dict_records, syn_idx

        ## deal with remain attrs
        ##
        if check_add_intersection_view:
            group = base_groups[-1]
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']
            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)
            tmp_dict = {idx:value for idx,value in tmp_dict.items() if value > 0}
            
            tmp_dict_records = list(tmp_dict.keys())
            syn_idx = 0
            while syn_idx < record_n:
                progress_count = progress_count + 1
                if progress_count%1000 == 0 and file.num_progress < num_progress_limit:
                    progress_count = 0
                    file.num_progress = file.num_progress + 1
                    file.save()
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                pick = random.sample(tmp_dict_records, 1)[0]
                for ele in range(len(group)):
                    synthetic_data[syn_idx][group[ele]] = pick[ele]
                syn_idx += 1
                tmp_dict[pick] = tmp_dict[pick] - 1
                if tmp_dict[pick] == 0:
                    del tmp_dict[pick]
                    tmp_dict_records.remove(pick)
                del pick
            del tmp_dict, group_name, split_path, dict_path, tmp_dict_records, syn_idx, group
    print('[INFO] combination is completed')
    ## start Heuristic algorithm
    ##
    pivot = base_groups[0]
    base_groups_idx = 1
    
    total_time = time.time()
    for idx in range(len(intersection_groups)):
        ## 1-1, find related views -> fixed by attribute_grouping with coarse grain
        ## 1-2, check common attributes between pivot and intersection (if there is no intersect then execute next pivot)
        ## 2, build same views
        ## 3, compute L1-norm with each one
        ## 4, calculate total L1-norm by weighted arithmetic average
        
        ## STEP [1] & [2]
        ##
        while len(group_common(base_groups[base_groups_idx], intersection_groups[idx])) == 0:
            progress_count = progress_count + 1
            if progress_count%1000 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            pivot = pivot + base_groups[base_groups_idx]
            base_groups_idx += 1
        
        steps = 1000000
        early_stop = 10000
        recurrent_L1 = 0
        view = intersection_groups[idx]
        view_name = '/'.join(view)
        
        ## try all cases - from low domain to high domain
        ##
        all_cases = find_cases(base_groups[base_groups_idx], view)
        all_cases, all_cases_domains, all_cases_domains_list = sort_by_domain(all_cases, attr_info)

        for case_idx in range(len(all_cases)):
            progress_count = progress_count + 1
            if progress_count%1000 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            ## build no_noisy_view (from synthetic data)
            ##
            no_noise_view = np.zeros(all_cases_domains[case_idx])
            support_data = pd.DataFrame(synthetic_data)
            marginal_count = support_data.groupby(all_cases[case_idx])[all_cases[case_idx][0]].count()
            marginal_count = marginal_count.to_dict()
            del support_data

            for event in marginal_count.keys():
                no_noise_view[index_of_cell(list(event), all_cases_domains_list[case_idx])] = marginal_count[event]
            del marginal_count

            ## build noisy_view
            ##
            ## find sorted positions
            ##
            sorted_pos = [sort_idx for sort_idx in range(len(view)) if view[sort_idx] not in all_cases[case_idx]]
            sorted_pos.reverse()
            noise_view =  np.zeros(all_cases_domains[case_idx])
            split_path = marginal_table_dict[view_name]['savePath']
            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)
            
            ## build new contingency table
            ##
            new_view = copy.deepcopy(view)
            for sort_pos in sorted_pos:
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
            for idx2 in tmp_dict.keys():
                noise_view[index_of_cell(list(idx2), all_cases_domains_list[case_idx])] = tmp_dict[idx2]
            del tmp_dict, new_view
            
            recurrent_L1 = np.linalg.norm(no_noise_view - noise_view, ord=1)
            time.sleep(0.2)
            print('=' * 90)
            print('task:', all_cases[case_idx])
            print('domain size:', all_cases_domains[case_idx])
            print('recurrent_L1:', recurrent_L1)
        
            ##  start swapping
            ##
            swap_count = 0
            same_count = 0
            common_attrs_inner = group_common(base_groups[base_groups_idx], all_cases[case_idx])

            no_update = 0
            for step in range(steps):
                progress_count = progress_count + 1
                if progress_count%1000 == 0 and file.num_progress < num_progress_limit:
                    progress_count = 0
                    file.num_progress = file.num_progress + 1
                    file.save()
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                swap = random.sample(range(0, record_n), 2)
                
                b1 = []
                b2 = []
                a1 = []
                a2 = []
                d1 = True
                d2 = True
                check_pos = 0
                for attr in all_cases[case_idx]:
                    b1.append(synthetic_data[swap[0]][attr])
                    b2.append(synthetic_data[swap[1]][attr])
                    if attr in common_attrs_inner:
                        a1.append(synthetic_data[swap[0]][attr])
                        a2.append(synthetic_data[swap[1]][attr])
                        d1 = d1 and (a1[check_pos] == a2[check_pos])
                    else:
                        a1.append(synthetic_data[swap[1]][attr])
                        a2.append(synthetic_data[swap[0]][attr])
                        d2 = d2 and (a1[check_pos] == a2[check_pos])
                    check_pos += 1

                if d1 or d2:
                    same_count += 1
                    continue						
                
                small_noise_view = np.zeros(4)
                small_no_noise_view = np.zeros(4)
                small_no_noise_view_after = np.zeros(4)
                selected_position = [b1, b2, a1, a2]
                for pos_idxx in range(4):
                    small_noise_view[pos_idxx] =  copy.deepcopy(noise_view[index_of_cell(selected_position[pos_idxx], all_cases_domains_list[case_idx])])

                if sum(small_noise_view) == 0: ## without improvement
                    del small_noise_view, small_no_noise_view, small_no_noise_view_after, selected_position
                    same_count += 1
                    continue
                
                for pos_idxx in range(4):
                    small_no_noise_view[pos_idxx] = copy.deepcopy(no_noise_view[index_of_cell(selected_position[pos_idxx], all_cases_domains_list[case_idx])])
                recurrent_little_L1 = np.linalg.norm(small_no_noise_view - small_noise_view, ord=1)
                
                small_no_noise_view[0] -= 1
                small_no_noise_view[1] -= 1
                small_no_noise_view[2] += 1
                small_no_noise_view[3] += 1	
                swapped_L1 = np.linalg.norm(small_no_noise_view - small_noise_view, ord=1)
                
                if swapped_L1 < recurrent_little_L1:
                    ## update it
                    no_noise_view[index_of_cell(b1, all_cases_domains_list[case_idx])] -= 1
                    no_noise_view[index_of_cell(b2, all_cases_domains_list[case_idx])] -= 1
                    no_noise_view[index_of_cell(a1, all_cases_domains_list[case_idx])] += 1
                    no_noise_view[index_of_cell(a2, all_cases_domains_list[case_idx])] += 1

                    for attr in base_groups[base_groups_idx]:
                        tmp = synthetic_data[swap[1]][attr]
                        synthetic_data[swap[1]][attr] = synthetic_data[swap[0]][attr]
                        synthetic_data[swap[0]][attr] = tmp
                        del tmp
                    swap_count += 1
                    no_update = 0
                elif swapped_L1 == recurrent_little_L1:
                        same_count += 1
                        no_update += 1
                        if no_update == early_stop:
                            print('early stop for step:', step)
                            del swapped_L1, recurrent_little_L1
                            del small_noise_view, small_no_noise_view, small_no_noise_view_after, selected_position	
                            break
                del swapped_L1, recurrent_little_L1
                del small_noise_view, small_no_noise_view, small_no_noise_view_after, selected_position
            
            time.sleep(0.2)
            recurrent_L1 = np.linalg.norm(no_noise_view - noise_view, ord=1)
            print('Last_L1:', recurrent_L1)
            del recurrent_L1, no_noise_view, noise_view, swap_count
        del view, view_name
        pivot = pivot + base_groups[base_groups_idx]
        base_groups_idx += 1
    print('[INFO] Swapping stage is completed')

    if (check_case == 1) and (not check_add_intersection_view):
        ## use determintistic method
        ##
        insert_view = base_groups[-1]	
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
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
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
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                ## build probabilistic list
                count_choices = np.array([bs_dict[cells] for cells in bs_dict_cells])
                count_choices = np.divide(count_choices, remain_count)
                pick_idx = np.random.choice(range(len(bs_dict_cells)), p = count_choices)
                
                start_pos_idx = 0
                for attr in insert_view:
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
    print('=' * 90)
    print('Total time:', time.time() - total_time)
    synthetic_data = pd.DataFrame(synthetic_data, columns = attr_name_list)
    synthetic_data.to_csv(path_tmp + '/' + directory_name + '_synthetic.csv', index=0)
    synthetic_length = len(synthetic_data)
    del attr_info, synthetic_data, target, base_groups_idx
    print('synthetic_length:', synthetic_length)