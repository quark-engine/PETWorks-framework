import pandas as pd
import numpy as np
import copy
import os
from django.conf import settings
from DPView.utils.functions import *
from general.function import ContentDetection
import swifter

# def value_dispatch(v, conditions):
# 	v_tag = 0
# 	for condition in conditions:
# 		if v > condition:
# 			v_tag += 1
# 		else:
# 			return v_tag
# 	return v_tag

def value_dispatch(v, conditions):
    v_tag = 0
    if ((conditions[0][0] == conditions[0][1]) and (v == conditions[0][1])) or (v < conditions[0][1]):
        return v_tag
    v_tag = 1
    distance = v - conditions[0][1]
    for condition in conditions[1:]:
        if v >= condition[0]:
            if ((condition[0] == condition[1]) and (v == condition[0])) or (v < condition[1]):
                return v_tag
            else:
                v_tag += 1
                distance = v - condition[1]
        elif (condition[0] - v) > distance:
            return (v_tag - 1)
        else:
            return v_tag

def recovery_value_int(v, conditions):
    return np.random.randint(conditions[v][0], conditions[v][1])

def recovery_value_float(v, conditions):
    return np.random.uniform(conditions[v][0], conditions[v][1])

def reconstruct(a):
    output = []
    for local_a in a:
        output.append(local_a[1])
    return output

def preprocess(username, csv_name, attr_basic_info):
    detection = ContentDetection()
    directory_name = csv_name.split(".")[-2]
    root_path = settings.DPVIEW_TEMP_ROOT + username + '/' + directory_name + '/'
    mypath = root_path + '/tmp'
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    del mypath

    raw_data = pd.read_csv(settings.UPLOAD_ROOT + 'DPView/' + username + '/' + directory_name + '/' + csv_name, low_memory=False, float_precision='round_trip', dtype=str)
    missing_attr = raw_data.isnull().any()
    
    attr_name = raw_data.keys().tolist()
    attr_info_max = {}
    specs_mapping = {}  ## used for recovery process with categorical attributes
    specs_mapping['single_attr'] = {}
    
    for attr in attr_name:
        attr_info_max[attr] = {}
        specs_mapping[attr] = {}
        
        column = raw_data.loc[:, attr].values.tolist()
        column = [temp for temp in column if str(temp) != 'nan']
        
        if detection.is_number(column):
            if detection.is_float(column):
                raw_data[attr] = raw_data[attr].astype('float64')
            else:
                raw_data[attr] = raw_data[attr].astype('float64').astype('Int64')
        
        if attr_basic_info[attr]['type'] == 'num':
            if raw_data[attr].dtypes == 'int64' or raw_data[attr].dtypes == 'Int64' or raw_data[attr].dtypes == 'int':
                specs_mapping[attr]['num-type'] = 'int'
            else:
                specs_mapping[attr]['num-type'] = 'float'
            #conditions = reconstruct(attr_basic_info[attr]['bucket'])
            conditions = copy.deepcopy(attr_basic_info[attr]['bucket'])
            if missing_attr[attr]:
                raw_data[attr].fillna(99999999, inplace = True)
                #conditions.append(99999999)
                conditions.append([99999999, 99999999])
                raw_data[attr] = raw_data[attr].swifter.apply(value_dispatch, args=(conditions,))
                attr_info_max[attr]['max'] = len(attr_basic_info[attr]['bucket'])
                specs_mapping[attr]['optional'] = 1
            else:
                raw_data[attr] = raw_data[attr].swifter.apply(value_dispatch, args=(conditions,))
                attr_info_max[attr]['max'] = len(attr_basic_info[attr]['bucket']) - 1
                specs_mapping[attr]['optional'] = 0
            print(attr, 'is bucketized with bucket_size:', attr_info_max[attr]['max'])
        elif (attr_basic_info[attr]['type'] == 'cat') or (attr_basic_info[attr]['type'] == 'num2cat'):
            specs_mapping[attr]['dict'] =  {}
            name_v = raw_data[attr].value_counts().index.tolist()
            max_v = len(name_v)
            specs_mapping[attr]['count'] =  max_v
            if missing_attr[attr]:
                raw_data[attr].fillna(max_v, inplace = True)
                specs_mapping[attr]['dict'][str(max_v)] = np.nan
                attr_info_max[attr]['max'] = max_v
                specs_mapping[attr]['optional'] = 1
            else:
                attr_info_max[attr]['max'] = max_v - 1
                specs_mapping[attr]['optional'] = 0
            float_v = False
            if raw_data[attr].dtypes == 'float64' or raw_data[attr].dtypes == 'float':
                float_v = True
            raw_data[attr] = raw_data[attr].astype('str')
            name_v = raw_data[attr].value_counts().index.tolist()
            if missing_attr[attr]:
                if (attr_basic_info[attr]['type'] == 'num2cat') and float_v:
                    name_v.remove(str(float(max_v)))
                else:
                    name_v.remove(str(max_v))
            max_v = len(name_v)
            name_v.sort()
            for idx in range(max_v):
                raw_data[attr].replace(name_v[idx], idx, inplace = True)
                specs_mapping[attr]['dict'][str(idx)] = name_v[idx]
            if (attr_basic_info[attr]['type'] == 'num2cat') and float_v:
                raw_data[attr] = raw_data[attr].astype('float')
            raw_data[attr] = raw_data[attr].astype('int')
            del name_v, max_v
            print(attr, 'is categorical with dict_size:', attr_info_max[attr]['max'])
        else:
            if len(raw_data[attr].value_counts().index.tolist()) == 0:
                specs_mapping['single_attr'][attr] = np.nan
            else:
                specs_mapping['single_attr'][attr] = raw_data[attr].value_counts().index.tolist()[0]
            del raw_data[attr], attr_info_max[attr]
            print(attr, 'is a singel-value attribute')
    store(attr_info_max, root_path + '/' + directory_name +'_attr_info_max')
    store(specs_mapping, root_path + '/' + directory_name + '-specs-mapping')
    raw_data.to_csv(root_path + '/' + directory_name + '_transform.csv', index=0)
    print()
    print('[Info] Save ----- completed')
    del raw_data, attr_name, attr_info_max, specs_mapping

def recover_data(csv_name, username, path_tmp, allocation_mode, non_neg_mode, privacy_budget_name, size_marginal_tables, num_marginal_tables, num_bucket, synthetic_num, attr_basic_info):
    directory_name = csv_name.split(".")[-2]
    raw_data = pd.read_csv(path_tmp + '/' + directory_name + '_synthetic.csv', low_memory=False)
    
    attr_name = attr_basic_info.keys()
    specs_mapping = load(settings.DPVIEW_TEMP_ROOT + username + '/' + directory_name + '/'  + directory_name + '-specs-mapping')
    for attr in attr_name:
        if attr_basic_info[attr]['type'] == 'num':
            conditions = copy.deepcopy(attr_basic_info[attr]['bucket'])
            if specs_mapping[attr]['num-type'] == 'int':
                conditions = [[condition[0], condition[1]+1] for condition in conditions]
                if specs_mapping[attr]['optional'] == 0:
                    raw_data[attr] = raw_data[attr].swifter.apply(recovery_value_int, args=(conditions,))
                else:
                    conditions.append([99999999, 100000000])
                    raw_data[attr] = raw_data[attr].swifter.apply(recovery_value_int, args=(conditions,))
                    raw_data[attr].replace(99999999, np.nan, inplace = True)
            elif specs_mapping[attr]['optional'] == 0:
                    raw_data[attr] = raw_data[attr].swifter.apply(recovery_value_float, args=(conditions,))
            else:
                conditions.append([99999999, 99999999])
                raw_data[attr] = raw_data[attr].swifter.apply(recovery_value_float, args=(conditions,))
                raw_data[attr].replace(99999999.0, np.nan, inplace = True)
        elif (attr_basic_info[attr]['type'] == 'cat') or (attr_basic_info[attr]['type'] == 'num2cat'):
            for idx in range(specs_mapping[attr]['count']):
                if idx in raw_data[attr]:
                    raw_data.loc[raw_data[attr] == idx, attr] = specs_mapping[attr]['dict'][str(idx)]
        else:
            raw_data[attr] = specs_mapping['single_attr'][attr]    
    raw_data = raw_data.reindex(columns = attr_name)
    print(raw_data.head())

    original_data = pd.read_csv(settings.UPLOAD_ROOT + 'DPView/' + username + '/' + directory_name + '/' + csv_name, low_memory=False, float_precision='round_trip')
    for attr in attr_name:
        raw_data[attr] = raw_data[attr].astype(original_data[attr].dtypes)
    output_path = settings.OUTPUT_ROOT + 'DPView/' + username + '/' + directory_name + '/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    raw_data.to_csv(output_path + directory_name + '_output' + '.csv', index=0, float_format='%g')
    print('\n[Info] Data transformation ----- completed')
    del raw_data, attr_name, original_data, specs_mapping