import random
import copy
import math
import os

import numpy as np

from django.utils.translation import gettext
from general.exception import BreakProgramException
from DPView.utils.functions import *
#from itertools import permutations, combinations
from tqdm import tqdm


def group_relation(last_group, pick_group):
    different = list(set(pick_group).difference(set(last_group)))
    return different

def group_common(groupA, groupB):
    common = list(set(groupA).intersection(set(groupB)))
    return common
    
def sort_by_topk_specific_attr(base_groups, attr_info, intersect_attr_num, specific_attr):
    domain_size = [0] * len(base_groups)
    output = []
    ## find the product of top-k
    for idx in range(len(base_groups)):
        top_k = []
        for attr in base_groups[idx][:intersect_attr_num]:
            top_k.append(attr_info[attr]['max'] + 1)
            top_k.sort(reverse = True)
        for attr in base_groups[idx][intersect_attr_num:]:
            tmp_top = (attr_info[attr]['max'] + 1)
            if tmp_top > top_k[-1]:
                top_k[-1] = tmp_top
            del tmp_top
        mul = 1
        for mul_v in top_k:
            mul *= mul_v
        domain_size[idx] = mul
        del top_k, mul
    terminated = False
    first_exe = True
    while (len(output) != len(base_groups)) and (not terminated):
        max_idx = [idx for idx, value in enumerate(domain_size) if value == max(domain_size)]
        pick_group = random.sample(max_idx, 1)[0]
        if first_exe:
            if specific_attr not in base_groups[pick_group]:
                terminated = True
            first_exe = False
        output.append(base_groups[pick_group])
        domain_size[pick_group] = 0
    del domain_size
    return output, terminated

def sort_by_topk(base_groups, attr_info, intersect_attr_num):
    domain_size = [0] * len(base_groups)
    output = []
    ## find the product of top-k
    for idx in range(len(base_groups)):
        top_k = []
        for attr in base_groups[idx][:intersect_attr_num]:
            top_k.append(attr_info[attr]['max'] + 1)
            top_k.sort(reverse = True)
        for attr in base_groups[idx][intersect_attr_num:]:
            tmp_top = (attr_info[attr]['max'] + 1)
            if tmp_top > top_k[-1]:
                top_k[-1] = tmp_top
            del tmp_top
        mul = 1
        for mul_v in top_k:
            mul *= mul_v
        domain_size[idx] = mul
        del top_k, mul
    while len(output) != len(base_groups):
        max_idx = [idx for idx, value in enumerate(domain_size) if value == max(domain_size)]
        pick_group = random.sample(max_idx, 1)[0]
        output.append(base_groups[pick_group])
        domain_size[pick_group] = 0
    del domain_size
    return output

def gen_intersection_no_padding(base_groups, attr_info, intersect_attr_num, intersect_attr_num_complement, maximal_base_views_domain):
    output = []
    generate_fail = False
    size_marginal_tables = len(base_groups[0])
    for current_group_idx in range(len(base_groups) - 1):
        if generate_fail:
            continue
        list_a = []
        list_b = []
        pick = []
        for attr in base_groups[current_group_idx]:
            list_a.append(attr_info[attr]['max'] + 1)
        for attr in base_groups[current_group_idx + 1]:
            list_b.append(attr_info[attr]['max'] + 1)
        list_a = np.array(list_a)
        list_b = np.array(list_b)
        product_a = 1
        product_b = 1

        for sel in range(intersect_attr_num):
            max_pos_ = np.where(list_a == np.max(list_a))[0]
            product_a *= np.max(list_a)
            max_pos_ = random.choice(max_pos_)
            pick.append(base_groups[current_group_idx][max_pos_])
            list_a[max_pos_] = -1
            del max_pos_

        for sel in range(intersect_attr_num):
            min_pos_ = np.where(list_b == np.min(list_b))[0]
            product_b *= np.min(list_b)
            min_pos_ = random.choice(min_pos_)
            pick.append(base_groups[current_group_idx + 1][min_pos_])
            list_b[min_pos_] = 99999
            del min_pos_
        ## reverse the value
        for sel in range(size_marginal_tables):
            if list_a[sel] == -1:
                list_a[sel] = 99999
        if intersect_attr_num_complement == 1:
            if product_a > product_b * np.min(list_b):
                min_pos_ = np.where(list_b == np.min(list_b))[0]
                min_pos_ = random.choice(min_pos_)
                pick.append(base_groups[current_group_idx + 1][min_pos_])
            else:
                min_pos_ = np.where(list_a == np.min(list_a))[0]
                min_pos_ = random.choice(min_pos_)
                pick.append(base_groups[current_group_idx][min_pos_])
            del min_pos_
        ## compute domain size
        tmp_view_domain_size = 1
        for attr in pick:
            tmp_view_domain_size *= (attr_info[attr]['max'] + 1)
        if tmp_view_domain_size > maximal_base_views_domain:
            generate_fail = True
            output = []
            del list_a, list_b, pick, product_a, product_b
            continue
        pick.sort()
        output.append(pick)
        del list_a, list_b, pick, product_a, product_b
    return output, generate_fail

def gen_latest_intersection_view(attr_name_copy, attr_info, tmp_max_domain, intersect_attr_num, intersect_attr_num_complement, latest_view):
    output = []
    generate_fail = False
    maximal_size = 999999999

    T = 1000
    Tmin = 10
    t = 0

    while T >= Tmin:
        for search_time in range(20):
            latest_view_copy = copy.deepcopy(latest_view)
            attr_name_copy_copy = copy.deepcopy(attr_name_copy)
            # select the attrs for intersection view
            pre_attrs = random.sample(attr_name_copy_copy, k = intersect_attr_num)
            post_attrs = random.sample(latest_view, k = intersect_attr_num)

            # compute domain size
            pre_size = 1
            post_size = 1
            for attr in pre_attrs:
                pre_size *= (attr_info[attr]['max'] + 1)
            for attr in post_attrs:
                post_size *= (attr_info[attr]['max'] + 1)
            total_size = pre_size*post_size

            if intersect_attr_num_complement == 0:
                if (total_size <= tmp_max_domain):
                    if (total_size < maximal_size):
                        comb = pre_attrs + post_attrs
                        comb.sort()
                        output = copy.deepcopy(comb)
                        maximal_size = total_size
                        del comb, total_size
                    elif (total_size == maximal_size):
                        prob = 0.5
                        if random.uniform(0, 1) < prob:
                            comb = pre_attrs + post_attrs
                            comb.sort()
                            output = copy.deepcopy(comb)
                            del comb, total_size
            elif (total_size < tmp_max_domain):
                latest_view_copy = [attr for attr in latest_view_copy if attr not in post_attrs] 
                select_post = random.sample(latest_view_copy, k = 1)[0]
                select_post_after = post_size * (attr_info[select_post]['max'] + 1)
                if (pre_size > select_post_after):
                    total_size *= (attr_info[select_post]['max'] + 1)
                    if (total_size <= tmp_max_domain):
                        if (total_size < maximal_size):
                            comb = pre_attrs + post_attrs
                            comb.append(select_post)
                            comb.sort()
                            output = copy.deepcopy(comb)
                            maximal_size = total_size
                            del comb, total_size
                        elif (total_size == maximal_size):
                            prob = 0.5
                            if random.uniform(0, 1) < prob:
                                comb = pre_attrs + post_attrs
                                comb.append(select_post)
                                comb.sort()
                                output = copy.deepcopy(comb)
                                del comb, total_size
                else:
                    attr_name_copy_copy = [attr for attr in attr_name_copy_copy if attr not in pre_attrs]
                    select_pre = random.sample(attr_name_copy_copy, k = 1)[0]
                    select_pre_after = pre_size * (attr_info[select_pre]['max'] + 1)
                    if (select_pre_after > post_size):
                        total_size *= (attr_info[select_pre]['max'] + 1)
                        if (total_size <= tmp_max_domain):
                            if (total_size < maximal_size):
                                comb = pre_attrs + post_attrs
                                comb.append(select_pre)
                                comb.sort()
                                output = copy.deepcopy(comb)
                                maximal_size = total_size
                                del comb, total_size
                            elif (total_size == maximal_size):
                                prob = 0.5
                                if random.uniform(0, 1) < prob:
                                    comb = pre_attrs + post_attrs
                                    comb.append(select_pre)
                                    comb.sort()
                                    output = copy.deepcopy(comb)
                                    del comb, total_size
            del latest_view_copy, attr_name_copy_copy, pre_attrs, post_attrs, pre_size, post_size
        t += 1
        T = 1000/(1 + t)
    if len(output) == 0:
        generate_fail = True
    else:
        output.sort()
    return output, maximal_size, generate_fail

def find_padding_view(latest_base_group, attr_info, maximal_base_views_domain, remaining_attrs, fra):
    attr_name = list(attr_info.keys())
    cand_attr_name = copy.deepcopy(latest_base_group)
    padding_num = len(latest_base_group) - len(remaining_attrs)

    given_domain = 1
    for attr in remaining_attrs:
        given_domain *= (attr_info[attr]['max'] + 1)

    output = []
    generate_fail = False
    maximal_size = 999999999

    T = 1000 * fra
    Tmin = 10
    t = 0

    while T >= Tmin:
        for search_time in range(20):
            padding_attrs = random.sample(cand_attr_name, k = padding_num)
            padding_domain = 1
            for attr in padding_attrs:
                padding_domain *= (attr_info[attr]['max'] + 1)
            padding_domain *= given_domain

            if padding_domain <= maximal_base_views_domain:
                for attr in remaining_attrs:
                    padding_attrs.append(attr)
                padding_attrs.sort()
                if padding_domain < maximal_size: 
                    ## accept it
                    output = copy.deepcopy(padding_attrs)
                    maximal_size = padding_domain
                elif padding_domain == maximal_size:
                    prob = 0.5
                    if random.uniform(0, 1) < prob:
                        output = copy.deepcopy(padding_attrs)
                        maximal_size = padding_domain
                    del prob
            del padding_domain, padding_attrs
        t += 1
        T = 1000/(1 + t)
    if len(output) == 0:
        generate_fail = True
    else:
        output.sort()
    return output, generate_fail

def generate_groupList(size_marginal_tables, temp_root, path_tmp, directory_name, privacy, record_n, labeled_attr, check_add_intersection_view, file, num_progress_limit):
    ## Generate base views and intersection views
    ##
    ## Target: lower maximal domain of base view as possible, and
    ##         larger maximal domain of padding view as possible
    ## Constraint: (1) domain of the partition came from pre-view should be larger than or equal to
    ##                 the domain domain of the partition came from post-view
    ##             (2) domain of intersection view must less than or equal to
    ##                 the maximal domain of base view
    progress_count = 0
    fra_threshold = 6
    show_search_info = True

    attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
    attr_name = list(attr_info.keys())

    if not os.path.isdir(path_tmp):
        os.makedirs(path_tmp)

    check_case = 0
    if len(attr_name) % size_marginal_tables != 0:
        check_case = 1  # incomplete splitting
    
    base_groups = []
    intersection_groups = []

    max_sum_domain = 999999999

    intersect_attr_num = size_marginal_tables // 2
    intersect_attr_num_complement = size_marginal_tables % 2

    fra = 1
    if size_marginal_tables > fra_threshold:
        mul_fra = size_marginal_tables - fra_threshold
        if mul_fra < 3:
            fra *= 10 ** mul_fra
        else:
            fra *= 200

    if check_case == 0:
        T = 1000*fra
        Tmin = 10
        t = 0

        while T >= Tmin:
            find_new_solution = False
            for search_time in range(10*fra):
                ## generates base views
                attr_name_copy = copy.deepcopy(attr_name)
                tmp_base_groups = []
                tmp_max_domain = 0
                while len(attr_name_copy) != 0:
                    progress_count = progress_count + 1
                    if progress_count%20 == 0 and file.num_progress < num_progress_limit:
                        progress_count = 0
                        file.num_progress = file.num_progress + 1
                        file.save()
                    if file.skip:
                        raise BreakProgramException(gettext('程式成功終止'))
                    
                    tmp_ = random.sample(attr_name_copy, k = size_marginal_tables)
                    tmp_.sort()
                    tmp_base_groups.append(tmp_)
                    attr_name_copy = [ele for ele in attr_name_copy if ele not in tmp_]
                    tmp_max_domain_size = 1
                    for attr in tmp_:
                        tmp_max_domain_size *= (attr_info[attr]['max'] + 1)
                    if tmp_max_domain_size > tmp_max_domain:
                        tmp_max_domain = tmp_max_domain_size
                    del tmp_, tmp_max_domain_size
                if labeled_attr in attr_name:
                    tmp_base_groups, terminated = sort_by_topk_specific_attr(tmp_base_groups, attr_info, intersect_attr_num, labeled_attr)
                    if terminated:
                        del tmp_base_groups, tmp_max_domain
                        continue
                else:
                    tmp_base_groups = sort_by_topk(tmp_base_groups, attr_info, intersect_attr_num)
                tmp_intersection_groups, generate_fail = gen_intersection_no_padding(tmp_base_groups, attr_info, intersect_attr_num, intersect_attr_num_complement, tmp_max_domain)
                if generate_fail:
                    del tmp_base_groups, tmp_intersection_groups, tmp_max_domain
                    continue
                ## compute sum domain
                sum_domain = 0
                for group in tmp_base_groups:
                    mul = 1
                    for attr in group:
                        mul *= (attr_info[attr]['max'] + 1)
                    sum_domain += mul
                    del mul
                for group in tmp_intersection_groups:
                    mul = 1
                    for attr in group:
                        mul *= (attr_info[attr]['max'] + 1)
                    sum_domain += mul
                    del mul
                ## evaluation
                if sum_domain < max_sum_domain: 
                    base_groups = copy.deepcopy(tmp_base_groups)
                    intersection_groups = copy.deepcopy(tmp_intersection_groups)
                    max_sum_domain = sum_domain
                elif sum_domain == max_sum_domain:
                    prob = 0.5
                    if random.uniform(0, 1) < prob:
                        base_groups = copy.deepcopy(tmp_base_groups)
                        intersection_groups = copy.deepcopy(tmp_intersection_groups)
                        max_sum_domain = sum_domain	
                    del prob
                del tmp_base_groups, tmp_intersection_groups, sum_domain
            t += 1
            if find_new_solution:
                if show_search_info:
                    print('%d time - %d' % (t, max_sum_domain))
            T = 1000*fra/(1 + t)
    else:
        remaining_size =  len(attr_name) % size_marginal_tables

        T = 1000*fra
        Tmin = 10
        t = 0

        while T >= Tmin:
            find_new_solution = False
            for search_time in range(10*fra):
                ## decide remaining attrs
                attr_name_copy = copy.deepcopy(attr_name)
                insert_labeled_attr = False
                if labeled_attr in attr_name:
                    insert_labeled_attr = True
                    attr_name_copy.remove(labeled_attr)
                remaining_attrs = random.sample(attr_name_copy, k = remaining_size)
                attr_name_copy = [ele for ele in attr_name_copy if ele not in remaining_attrs]
                ## generate base views
                tmp_base_groups = []
                tmp_max_domain = 0
                sum_domain = 0
                while len(attr_name_copy) != 0:
                    progress_count = progress_count + 1
                    if progress_count%20 == 0 and file.num_progress < num_progress_limit:
                        progress_count = 0
                        file.num_progress = file.num_progress + 1
                        file.save()
                    if file.skip:
                        raise BreakProgramException(gettext('程式成功終止'))
                    if insert_labeled_attr:
                        tmp_ = random.sample(attr_name_copy, k = size_marginal_tables - 1)
                        tmp_.append(labeled_attr)
                        insert_labeled_attr = False
                    else:					
                        tmp_ = random.sample(attr_name_copy, k = size_marginal_tables)
                    tmp_.sort()
                    tmp_base_groups.append(tmp_)
                    attr_name_copy = [ele for ele in attr_name_copy if ele not in tmp_]
                    tmp_max_domain_size = 1
                    for attr in tmp_:
                        tmp_max_domain_size *= (attr_info[attr]['max'] + 1)
                    if tmp_max_domain_size > tmp_max_domain:
                        tmp_max_domain = tmp_max_domain_size
                    del tmp_, tmp_max_domain_size
                if labeled_attr in attr_name:
                    tmp_base_groups, terminated = sort_by_topk_specific_attr(tmp_base_groups, attr_info, intersect_attr_num, labeled_attr)
                    if terminated:
                        del tmp_base_groups, tmp_max_domain
                        contin
                else:
                    tmp_base_groups = sort_by_topk(tmp_base_groups, attr_info, intersect_attr_num)
                if check_add_intersection_view:
                    ## (I) generating the latest base view without padding
                    latest_view = copy.deepcopy(remaining_attrs)
                    latest_view.sort()				
                else:
                    latest_view, generate_fail = find_padding_view(tmp_base_groups[-1], attr_info, tmp_max_domain, remaining_attrs, fra)
                    if generate_fail:
                        del attr_name_copy, remaining_attrs, tmp_base_groups, tmp_max_domain
                        continue
                tmp_intersection_groups, generate_fail = gen_intersection_no_padding(tmp_base_groups, attr_info, intersect_attr_num, intersect_attr_num_complement, tmp_max_domain)
                if generate_fail:
                    del tmp_base_groups, tmp_intersection_groups, tmp_max_domain
                    continue
                if check_add_intersection_view:
                    ## (II) generating the latest intersection view
                    attr_name_copy = copy.deepcopy(attr_name)
                    attr_name_copy = [ele for ele in attr_name_copy if ele not in latest_view]
                    
                    latest_intersection_view, check_size, generate_fail = gen_latest_intersection_view(attr_name_copy, attr_info, tmp_max_domain, intersect_attr_num, intersect_attr_num_complement, latest_view)				
                    if generate_fail:
                        del tmp_base_groups, tmp_intersection_groups, tmp_max_domain, attr_name_copy
                        continue

                ## compute sum domain
                sum_domain = 0
                for group in tmp_base_groups:
                    mul = 1
                    for attr in group:
                        mul *= (attr_info[attr]['max'] + 1)
                    sum_domain += mul
                    del mul
                for group in tmp_intersection_groups:
                    mul = 1
                    for attr in group:
                        mul *= (attr_info[attr]['max'] + 1)
                    sum_domain += mul
                    del mul
                mul = 1
                for attr in latest_view:
                    mul *= (attr_info[attr]['max'] + 1)
                sum_domain += mul
                del mul

                if check_add_intersection_view:
                    sum_domain += check_size
                ## evaluation
                if sum_domain < max_sum_domain: 
                    base_groups = copy.deepcopy(tmp_base_groups)
                    base_groups.append(latest_view)
                    intersection_groups = copy.deepcopy(tmp_intersection_groups)
                    if check_add_intersection_view:
                        intersection_groups.append(latest_intersection_view)
                    max_sum_domain = sum_domain
                elif sum_domain == max_sum_domain: 
                    prob = 0.5
                    if random.uniform(0, 1) < prob:
                        base_groups = copy.deepcopy(tmp_base_groups)
                        base_groups.append(latest_view)
                        intersection_groups = copy.deepcopy(tmp_intersection_groups)
                        if check_add_intersection_view:
                            intersection_groups.append(latest_intersection_view)
                        max_sum_domain = sum_domain	
                    del prob
                del tmp_base_groups, tmp_intersection_groups, sum_domain, attr_name_copy

                if check_add_intersection_view:
                    del latest_intersection_view, check_size
            t += 1
            if find_new_solution:
                if show_search_info:
                    print('%d time - %d' % (t, max_sum_domain))
            T = 1000*fra/(1 + t)

    attribute_groups = base_groups + intersection_groups

    print('-' * 90)
    print('\n')

    total_domain = 0
    print('[Info] Numbers of marginal_group: %d' % len(attribute_groups))
    print('[Info] Generate marginal_group list ----- success\n\n')
    print('base_groups:')
    for group in base_groups:
        attr_sizess = []
        print(group)
        for attr in group:
            attr_sizess.append(attr_info[attr]['max'] + 1)
        tmp_prod = np.prod(attr_sizess)
        print(attr_sizess, tmp_prod)
        total_domain += tmp_prod
        del attr_sizess, tmp_prod
    print('-' * 90)
    print('intersection_groups:')
    for group in intersection_groups:
        attr_sizess = []
        print(group)
        for attr in group:
            attr_sizess.append(attr_info[attr]['max'] + 1)
        tmp_prod = np.prod(attr_sizess)
        print(attr_sizess, tmp_prod)
        total_domain += tmp_prod
        del attr_sizess, tmp_prod

    store(attribute_groups, path_tmp + '/marginal_group')
    del attribute_groups, attr_info, attr_name, total_domain
    base_intersection_collect = [base_groups, intersection_groups]
    store(base_intersection_collect, path_tmp + '/base_intersection_collect')
    del base_intersection_collect, base_groups, intersection_groups