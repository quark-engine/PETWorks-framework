#!/usr/bin/env python
# coding: utf-8

# # <font color=#00BFFF>Synthesize data</font>
# <br/>

# In[1]:

from django.http import JsonResponse
from django.conf import settings
from django.utils.translation import gettext
from general.exception import BreakProgramException
from general.models import ExecuteModel

import pandas as pd
import random
import json
import time
import os
import shutil

from DPView.utils import load_data_user_mode
from DPView.utils import check_zero
from DPView.utils import attribute_grouping_sum_domain
from DPView.utils import build_marginal
from DPView.utils import empirical_count
from DPView.utils import empty_consistency
from DPView.utils import non_negativity
from DPView.utils import consistency_0627
from DPView.utils import normalization_0419
from DPView.utils import synthetic_data_new_0510
from DPView.utils import synthetic_data_new_random_M

# 0510 -> determinatistic method
# random_M -> probabilistic method

import datetime
from datetime import date 
today = date.today()
import logging
logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s : %(message)s',datefmt='%Y-%m-%d %H:%M:%S',filename= str(today) +'_log.txt')

# In[2]:

def show_progress(request):
    try:
        username = request.user.get_username()
        file = ExecuteModel.objects.get(user_name=username)
        data = {
            'log':file.log,
            'num_progress':file.num_progress,
        }
        return JsonResponse(data, safe=False)
    except:
        data = {
            'log':gettext('程式已被取消執行，請重新嘗試執行'),
            'num_progress':0,
        }
        return JsonResponse(data, safe=False)
        
def break_program(file):
    file.skip = True
    file.save()
    
def run(request):
    username = request.user.get_username()
    file = ExecuteModel.objects.get(user_name=username)

    logging.info('##################################################\tDPView Start\t##################################################\n')

    csv_name = str(request.GET.get('csv_name', None))
    directory_name = csv_name.split(".")[-2]
    username = request.user.get_username()
    
    privacy_budget = float(request.GET.get('privacy_budget', None))      # float, need "point" symbol
    delta = 1.e-5             # if the value is -1, we use 1/n where n is records in data
    synthetic_num = -1        # -1 implies that the size as same as dataset
    num_bucket = int(request.GET.get('num_bucket', None))

    file.num_progress = 1
    file.log = 'Building data information...'
    file.save()
    logging.info('----------------------------------------\tBuild data information START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))
    
    dict_file_name = settings.UPLOAD_ROOT + 'DPView/' + username + '/' + directory_name + '/' + directory_name + '_dict.json'
    with open(dict_file_name) as json_file:
        attr_basic_info = json.load(json_file)
        
    '''
    attr_basic_info = {'age':{'type':'num', 'bucket':[[16, 50], [51, 100]]},
                      ' workclass':{'type':'cat'},
                      ' fnlwgt':{'type':'num', 'bucket':[[12000, 756000], [756001, 1500000]]},
                      ' education':{'type':'cat'},
                      ' education-num':{'type':'num2cat'},
                      ' marital-status':{'type':'cat'},
                      ' occupation':{'type':'cat'},
                      ' relationship':{'type':'cat'},
                      ' race':{'type':'cat'},
                      ' sex':{'type':'cat'},
                      ' capital-gain':{'type':'num', 'bucket':[[0, 0], [100, 20655], [20656, 41310], [999999, 999999]]},
                      ' capital-loss':{'type':'num', 'bucket':[[0, 0], [150, 2250], [2251, 4500]]},
                      ' hours-per-week':{'type':'num', 'bucket':[[1, 50], [51, 99]]},
                      ' native-country':{'type':'cat'},
                      ' class':{'type':'cat'}}

                      ## for 'type':'single', it should be removed at the first
                      
    attr_basic_info = {'Age':{'type':'num', 'bucket':[[0,50], [50,100]]},
                  'Department':{'type':'cat'},
                  'Salary':{'type':'cat'}}
    '''

    file.num_progress = 3
    file.log = 'Synthesizing data...'
    file.save()
    logging.info('----------------------------------------\tSynthesize data START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))
        
    size_views = 3            # depend on dataset

    allocation_mode = 1       # stragies{0: uniform allocation,
                              #          1: based on views' domain}

    non_neg_mode = 3          # stragies{0: minMax method,
                              #          1: ripple method,
                              #          2: directly remove negative number,
                              #          3: neighboring ripple method}
                              #          4: do nothing}
    labeled_attr = ''


    syn_mechanism = 0         # stragies{0: deterministic method (for small data),
                              #          1: probabilistic method (for large data)}

    ###################[ Used for administer ]#######################
    pass_the_preprocess = False
    pass_the_marginal_group = False
    
    logging.info('It cost %f sec' % (time.time() - tStart))
    logging.info('----------------------------------------\tSynthesize data END\t----------------------------------------\n')

    # In[3]:


    ## settings for numerical attribute
        ## categorical attr (directly transform into categorical type)
        ## otherwise (transform into categorical type, need special bucketization)

    ## setting for categorical attribute
        ## ordinal attr should be transformed into numerical attr

    file.num_progress = 5
    file.log = 'Preprocessing and collecting data info...'
    file.save()
    logging.info('----------------------------------------\tPreprocess and collect data info START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))

    # In[5]:
    if not pass_the_preprocess:
        logging.info('Start preprocessing .....\n')
        tStart = time.time()
        load_data_user_mode.preprocess(username, csv_name, attr_basic_info)
        logging.info('Load transformed data ----- success')
        logging.info('It cost %f sec' % (time.time() - tStart))

    # ## <font color=#00BFFF>Preprocess and collect data info.</font>
    # <br/>

    # In[4]:

    raw_data = pd.read_csv(settings.DPVIEW_TEMP_ROOT + username + '/' + directory_name + '/' + directory_name + '_transform.csv', low_memory=False, float_precision='round_trip')
    attr_name = raw_data.keys().tolist()
    record_n, _ = raw_data.shape
    if synthetic_num <= 0:
        synthetic_num = record_n
    del raw_data

    degenerate2flat = False
    
    if len(attr_name) > size_views:
        num_views = len(attr_name) // size_views

        check_case = 0
        if len(attr_name) % size_views != 0:
            check_case = 1  # incomplete splitting

        check_add_intersection_view = False
        if check_case == 1:
            intersect_attr_num = size_views // 2
            intersect_attr_num_complement = size_views % 2
            
            if (len(attr_name) % size_views) >= (intersect_attr_num + 1):
                check_add_intersection_view = True
                num_views = 2 * num_views + 1
            else:
                num_views = 2 * num_views
        else:
            num_views = 2 * num_views - 1
        logging.info('check_add_intersection_view: %r' % check_add_intersection_view)
    else:
        degenerate2flat = True
        num_views = 1
        logging.info('degenerate to flat vector method')    

    # ## <font color=#00BFFF>Group marginal set</font>
    # <br/>

    # In[6]:
    file.num_progress = 10
    file.log = 'Grouping marginal set...'
    file.save()
    logging.info('----------------------------------------\tGroup marginal set START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))
    
    temp_root = settings.DPVIEW_TEMP_ROOT + username + '/'
    privacy_budget_name = str(privacy_budget).split('.')[0] + '_' + str(privacy_budget).split('.')[1]
    path_tmp = temp_root + directory_name + '/tmp/' + directory_name + '_' + privacy_budget_name + '_am_' + str(allocation_mode) + '_ngm_' + str(non_neg_mode) + '_'  + str(size_views) + '_' + str(num_views) 

    if not pass_the_marginal_group:
        logging.info('Start generate_groupList .....\n')
        tStart = time.time()
        if not degenerate2flat:
            attribute_grouping_sum_domain.generate_groupList(size_views, temp_root, path_tmp, directory_name, privacy_budget, record_n, labeled_attr, check_add_intersection_view, file, 20) 
        logging.info('generate_groupList ----- success')
        logging.info('It cost %f sec' % (time.time() - tStart))


    # ## <font color=#00BFFF>Generate marginal table</font>
    # <br/>

    # In[7]:
    file.num_progress = 20
    file.log = 'Generating marginal table...'
    file.save()
    logging.info('----------------------------------------\tGenerate marginal table START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))
    if not degenerate2flat:
        build_marginal.build_all_marginal(privacy_budget, temp_root, path_tmp, directory_name, num_views, allocation_mode)
    else:
        build_marginal.build_marginal(privacy_budget, temp_root, path_tmp, directory_name, attr_name)
    logging.info('It cost %f sec' % (time.time() - tStart))
    logging.info('----------------------------------------\tGenerate marginal table END\t----------------------------------------\n')

    # ## <font color=#00BFFF>Count empirical numbers</font>
    # <br/>

    # In[8]:
    file.num_progress = 25
    file.log = 'Generating marginal empirical table...'
    file.save()
    logging.info('----------------------------------------\tGenerate marginal empirical table START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))
    if not degenerate2flat:    
        empirical_count.empirical_counting(directory_name, temp_root, path_tmp)
    else:
        empirical_count.empirical_counting_flat_vector(directory_name, temp_root, path_tmp, attr_name)
    logging.info('Generate marginal empirical table ----- success')
    logging.info('It cost %f sec' % (time.time() - tStart))

    # In[9]:
    '''
    ###################[ Used for administer ]#######################
    if not degenerate2flat:
        check_zero.check_values(num_views, path_tmp, record_n)
    '''

    # ## <font color=#00BFFF>Consistency process</font>
    # <br/>

    # In[10]:
    file.num_progress = 30
    file.save()
    logging.info('----------------------------------------\tNon-negativity process START\t----------------------------------------')
    logging.info('----------------------------------------\tConsistency process START\t----------------------------------------')
    tStart = time.time()
    if not degenerate2flat:
        num_progress_limit = 30
        for run_t in range(3):
            if file.skip:
                raise BreakProgramException(gettext('程式成功終止'))
            file.log = 'Non-negativity processing...'
            file.save()
            empty_consistency.empty_consistency(size_views, directory_name, temp_root, path_tmp, non_neg_mode)
            num_progress_limit = num_progress_limit + 2
            file.log = 'Consistency processing...'
            file.save()
            consistency_0627.consistency(size_views, directory_name, temp_root, path_tmp, non_neg_mode, check_add_intersection_view, file, num_progress_limit)
            num_progress_limit = num_progress_limit + 2
            file.log = 'Non-negativity processing...'
            file.save()
            non_negativity.non_Negativity(num_views, directory_name, temp_root, path_tmp, non_neg_mode, file, num_progress_limit)
    else:
        non_negativity.non_Negativity_flat(directory_name, temp_root, path_tmp, attr_name, non_neg_mode, file, 60)
    logging.info('Consistency process ----- success')
    logging.info('It cost %f sec' % (time.time() - tStart))

    # In[11]:
    '''
    ###################[ Used for administer ]#######################
    if not degenerate2flat:
        check_values(num_views, path_tmp, record_n)
    '''
    
    # ## <font color=#00BFFF>Normalize marginal tables.</font>
    # <br/>

    # In[12]:
    file.num_progress = 42
    file.log = 'Normalization processing...'
    file.save()
    logging.info('----------------------------------------\tNormalization process START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))
    if not degenerate2flat:
        normalization_0419.normalization_tmp1(path_tmp, synthetic_num)
        normalization_0419.normalization(path_tmp, attr_name)
    else:
        random_gen = normalization_0419.normalization_flat(path_tmp, synthetic_num, attr_name)
    logging.info('Normalization process ----- success')
    logging.info('It cost %f sec' % (time.time() - tStart))

    # ## <font color=#00BFFF>Synthesize data with marginal tables.</font>
    # <br/>

    # In[13]:
    file.num_progress = 65
    file.log = 'Synthesizing data with marginal tables...'
    file.save()
    logging.info('----------------------------------------\tSynthesize data with marginal tables START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))

    if not degenerate2flat:
        if syn_mechanism == 0:
            synthetic_data_new_0510.synthesize_data(size_views, num_views, synthetic_num, directory_name, temp_root, path_tmp, attr_name, check_add_intersection_view, file, 95)
        else:
            synthetic_data_new_random_M.synthesize_data_random(size_views, num_views, synthetic_num, directory_name, temp_root, path_tmp, attr_name, check_add_intersection_view, file, 95)
    else:
        synthetic_data_new_0510.synthesize_data_flat(directory_name, temp_root, path_tmp, attr_name, record_n, random_gen, file, 95)
        
    logging.info('It cost %f sec' % (time.time() - tStart))
    logging.info('----------------------------------------\tSynthesize data with marginal tables END\t----------------------------------------\n')


    # ## <font color=#00BFFF>Data transformation.</font>
    # <br/>

    # In[14]:
    file.num_progress = 95
    file.log = 'Data transformation...'
    file.save()
    logging.info('----------------------------------------\tData transformation START\t----------------------------------------')
    tStart = time.time()
    if file.skip:
        raise BreakProgramException(gettext('程式成功終止'))

    load_data_user_mode.recover_data(csv_name, username, path_tmp, allocation_mode, non_neg_mode, privacy_budget_name, size_views, num_views, num_bucket, synthetic_num, attr_basic_info)

    file.num_progress = 100
    file.log = 'DPView Complete'
    file.save()
    logging.info('It cost %f sec' % (time.time() - tStart))
    logging.info('----------------------------------------\tData transformation END\t----------------------------------------\n')

    logging.info('##################################################\tDPView END\t##################################################\n\n\n')

    shutil.rmtree(settings.DPVIEW_TEMP_ROOT + username + '/' + directory_name)