import numpy as np
import copy
import os

from django.utils.translation import gettext
from general.exception import BreakProgramException
from DPView.utils.functions import *
from tqdm import tqdm


def find_neighboring_set(a, domains):
    output = []
    for pos in range(len(a)):
        for val in range(0,a[pos],1):
            output.append(a[:pos]+(val,)+a[pos+1:])
        for val in range(a[pos]+1,domains[pos],1):
            output.append(a[:pos]+(val,)+a[pos+1:])
    return output

def non_Negativity(num_marginal_tables, directory_name, temp_root, path_tmp, mode, file, num_progress_limit):
    print('[Info] Start non-negative post-processing .....\n')
    marginal_table_dict = load(path_tmp + '/marginal_table_dictionary')
    marginal_group_subValue = pickle_load(path_tmp + '/marginal_group_sub_num')
    marginal_group = load(path_tmp + '/marginal_group')
    progress_count = 0
    
    if mode == 0:
        for group in tqdm(marginal_group, ncols = 90):
            progress_count = progress_count + 1
            if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']
            #sub_num = marginal_group_subValue[group_name]['minValue']

            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)

            sub_num = tmp_dict[min(tmp_dict, key=tmp_dict.get)]

            tmp_dict = {key: tmp_dict[key] - sub_num for key in tmp_dict.keys()}
            pickle_store(tmp_dict, dict_path)
            del tmp_dict
    elif mode == 1:
        for group in tqdm(marginal_group, ncols = 90):
            progress_count = progress_count + 1
            if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']
            # sub_num = marginal_group_subValue[group_name]['negSum']
            # total_num = marginal_group_subValue[group_name]['posSum']

            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)

            sub_num = abs(sum(value for _, value in tmp_dict.items() if value < 0))
            total_num = sum(value for _, value in tmp_dict.items() if value > 0)

            if total_num < sub_num:
                tmp_dict = dict.fromkeys(tmp_dict, 0)
            else:
                tmp_dict_neg = {idx:0 for idx,value in tmp_dict.items() if value < 0}
                tmp_dict.update(tmp_dict_neg)
                del tmp_dict_neg
                tmp_dict_pos = {idx:value for idx,value in tmp_dict.items() if value > 0}
                tmp_dict_pos = sorted(tmp_dict_pos.items(), key=lambda d: d[1])
                tmp_dict_pos_value = np.array([ele[1] for ele in tmp_dict_pos])
                tmp_dict_pos_value = np.cumsum(tmp_dict_pos_value)
                tmp_dict_pos_zero = np.where(tmp_dict_pos_value < sub_num)[0]
                if len(tmp_dict_pos_zero) > 0:
                    tmp_dict_pos_zero = tmp_dict_pos_zero[-1]
                else:
                    tmp_dict_pos_zero = -1
                    print(tmp_dict_pos_value[0], ' with sub: ', sub_num)

                remain_value = tmp_dict_pos_value[tmp_dict_pos_zero + 1] - sub_num
                tmp_dict.update({tmp_dict_pos[tmp_dict_pos_zero + 1][0]:remain_value})
                del remain_value
                tmp_dict_zero = {ele[0]:0 for ele in tmp_dict_pos[:tmp_dict_pos_zero + 1]}
                tmp_dict.update(tmp_dict_zero)
                del tmp_dict_zero, tmp_dict_pos, tmp_dict_pos_zero
            pickle_store(tmp_dict, dict_path)
            del tmp_dict
    elif mode == 2:
        for group in tqdm(marginal_group, ncols = 90):
            progress_count = progress_count + 1
            if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']

            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)
            tmp_dict_zero = {idx:0 for idx,value in tmp_dict.items() if value < 0}
            tmp_dict.update(tmp_dict_zero)
            pickle_store(tmp_dict, dict_path)
            del tmp_dict, tmp_dict_zero, dict_path, split_path, group_name
    elif mode == 3:
        attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
        for group in tqdm(marginal_group, ncols = 90):
            progress_count = progress_count + 1
            if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                progress_count = 0
                file.num_progress = file.num_progress + 1
                file.save()
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            group_name = '/'.join(group)
            split_path = marginal_table_dict[group_name]['savePath']
            # sub_num = marginal_group_subValue[group_name]['negSum']
            # total_num = marginal_group_subValue[group_name]['posSum']

            dict_path = path_tmp + '/marginal_tables/marginal_table_' + str(split_path)
            tmp_dict = pickle_load(dict_path)

            sub_num = abs(sum(value for _, value in tmp_dict.items() if value < 0))
            total_num = sum(value for _, value in tmp_dict.items() if value > 0)

            if total_num < sub_num:
                tmp_dict = dict.fromkeys(tmp_dict, 0)
            elif total_num > sub_num:
                divisor = 0
                domains = []
                for attr in group:
                    divisor += attr_info[attr]['max']
                    domains.append(attr_info[attr]['max'] + 1)

                neg_cells = [cell for cell, value in tmp_dict.items() if value < 0]

                while len(neg_cells) > 0:
                    progress_count = progress_count + 1
                    if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                        progress_count = 0
                        file.num_progress = file.num_progress + 1
                        file.save()
                    if file.skip:
                        raise BreakProgramException(gettext('程式成功終止'))
                    ## find all cells that with negative count
                    for cell in neg_cells:
                        ## list all neighboring cells
                        nerghbors = find_neighboring_set(cell, domains)
                        local_sub_num = tmp_dict[cell]/divisor
                        for neighbor in nerghbors:
                            tmp_dict[neighbor] += local_sub_num
                        tmp_dict[cell] = 0
                        del local_sub_num, nerghbors
                    ## filtering small count
                    neg_cells = [cell for cell, value in tmp_dict.items() if value < 0]
                    tmp_dict_zero = {cell:0 for cell in neg_cells if tmp_dict[cell] > -1.e-16}
                    if len(tmp_dict_zero) > 0:
                        progress_count = progress_count + 1
                        if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                            progress_count = 0
                            file.num_progress = file.num_progress + 1
                            file.save()
                        if file.skip:
                            raise BreakProgramException(gettext('程式成功終止'))
                        tmp_dict.update(tmp_dict_zero)
                        for cell in list(tmp_dict_zero.keys()):
                            neg_cells.remove(cell)
                    del tmp_dict_zero
                del divisor, domains
            pickle_store(tmp_dict, dict_path)
            del tmp_dict, dict_path, split_path, group_name, sub_num, total_num
    else:
        do_nothing = True
    del marginal_table_dict, marginal_group_subValue, marginal_group
    
def non_Negativity_flat(directory_name, temp_root, path_tmp, group, mode, file, num_progress_limit):
    print('[Info] Start non-negative post-processing .....\n')
    progress_count = 0
    
    if mode == 0:
        dict_path = path_tmp + '/marginal_tables/marginal_table'
        tmp_dict = pickle_load(dict_path)
        sub_num = tmp_dict[min(tmp_dict, key=tmp_dict.get)]
        tmp_dict = {key: tmp_dict[key] - sub_num for key in tmp_dict.keys()}
        pickle_store(tmp_dict, dict_path)
        del tmp_dict
    elif mode == 1:
        dict_path = path_tmp + '/marginal_tables/marginal_table'
        tmp_dict = pickle_load(dict_path)

        sub_num = abs(sum(value for _, value in tmp_dict.items() if value < 0))
        total_num = sum(value for _, value in tmp_dict.items() if value > 0)

        if total_num < sub_num:
            tmp_dict = dict.fromkeys(tmp_dict, 0)
        else:
            tmp_dict_neg = {idx:0 for idx,value in tmp_dict.items() if value < 0}
            tmp_dict.update(tmp_dict_neg)
            del tmp_dict_neg
            tmp_dict_pos = {idx:value for idx,value in tmp_dict.items() if value > 0}
            tmp_dict_pos = sorted(tmp_dict_pos.items(), key=lambda d: d[1])
            tmp_dict_pos_value = np.array([ele[1] for ele in tmp_dict_pos])
            tmp_dict_pos_value = np.cumsum(tmp_dict_pos_value)
            tmp_dict_pos_zero = np.where(tmp_dict_pos_value < sub_num)[0]
            if len(tmp_dict_pos_zero) > 0:
                tmp_dict_pos_zero = tmp_dict_pos_zero[-1]
            else:
                tmp_dict_pos_zero = -1
                print(tmp_dict_pos_value[0], ' with sub: ', sub_num)

            remain_value = tmp_dict_pos_value[tmp_dict_pos_zero + 1] - sub_num
            tmp_dict.update({tmp_dict_pos[tmp_dict_pos_zero + 1][0]:remain_value})
            del remain_value
            tmp_dict_zero = {ele[0]:0 for ele in tmp_dict_pos[:tmp_dict_pos_zero + 1]}
            tmp_dict.update(tmp_dict_zero)
            del tmp_dict_zero, tmp_dict_pos, tmp_dict_pos_zero
        pickle_store(tmp_dict, dict_path)
        del tmp_dict
    elif mode == 2:
        dict_path = path_tmp + '/marginal_tables/marginal_table'
        tmp_dict = pickle_load(dict_path)
        tmp_dict_zero = {idx:0 for idx,value in tmp_dict.items() if value < 0}
        tmp_dict.update(tmp_dict_zero)
        pickle_store(tmp_dict, dict_path)
        del tmp_dict, tmp_dict_zero, dict_path
    elif mode == 3:
        attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
        dict_path = path_tmp + '/marginal_tables/marginal_table'
        tmp_dict = pickle_load(dict_path)

        sub_num = abs(sum(value for _, value in tmp_dict.items() if value < 0))
        total_num = sum(value for _, value in tmp_dict.items() if value > 0)

        if total_num < sub_num:
            tmp_dict = dict.fromkeys(tmp_dict, 0)
        elif total_num > sub_num:
            divisor = 0
            domains = []
            for attr in group:
                divisor += attr_info[attr]['max']
                domains.append(attr_info[attr]['max'] + 1)

            neg_cells = [cell for cell, value in tmp_dict.items() if value < 0]

            while len(neg_cells) > 0:
                progress_count = progress_count + 1
                if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                    progress_count = 0
                    file.num_progress = file.num_progress + 1
                    file.save()
                if file.skip:
                    raise BreakProgramException(gettext('程式成功終止'))
                ## find all cells that with negative count
                for cell in neg_cells:
                    ## list all neighboring cells
                    nerghbors = find_neighboring_set(cell, domains)
                    local_sub_num = tmp_dict[cell]/divisor
                    for neighbor in nerghbors:
                        tmp_dict[neighbor] += local_sub_num
                    tmp_dict[cell] = 0
                    del local_sub_num, nerghbors
                ## filtering small count
                neg_cells = [cell for cell, value in tmp_dict.items() if value < 0]
                tmp_dict_zero = {cell:0 for cell in neg_cells if tmp_dict[cell] > -1.e-16}
                if len(tmp_dict_zero) > 0:
                    progress_count = progress_count + 1
                    if progress_count%200 == 0 and file.num_progress < num_progress_limit:
                        progress_count = 0
                        file.num_progress = file.num_progress + 1
                        file.save()
                    if file.skip:
                        raise BreakProgramException(gettext('程式成功終止'))
                    tmp_dict.update(tmp_dict_zero)
                    for cell in list(tmp_dict_zero.keys()):
                        neg_cells.remove(cell)
                del tmp_dict_zero
            del divisor, domains
        pickle_store(tmp_dict, dict_path)
        del tmp_dict, dict_path, sub_num, total_num
    else:
        do_nothing = True