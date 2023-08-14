import numpy as np
import itertools
import copy
import math
import os

from DPView.utils.functions import *
from scipy.special import comb
from tqdm import tqdm


def build_all_marginal(privacy_budget, temp_root, path_tmp, directory_name, num_marginal_tables, allocation_mode):

	mypath = path_tmp + '/marginal_tables'
	if not os.path.isdir(mypath):
		os.makedirs(mypath)

	attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
	marginal_group = load(path_tmp + '/marginal_group')
	laplace_scale = np.ones(num_marginal_tables)
	pi_values = np.ones(num_marginal_tables)
	
	## mode-0: uniform allocation
	## mode-1: based on views' domain
	##
	if allocation_mode == 0:
		avg_scale = num_marginal_tables/privacy_budget
		laplace_scale = [avg_scale for value in laplace_scale]
		pi_values = [1/num_marginal_tables for value in pi_values]
	else:
		views_domain = []
		for view in marginal_group:
			mul = 1
			for attr in view:
				mul *= (attr_info[attr]['max'] + 1)
			views_domain.append(mul)
			del mul
		div_weight = 0
		for ci in views_domain:
			div_weight += ci**(1/3)
		div_weight = div_weight**3
		views_domain = np.array(views_domain)
		views_domain = np.divide(views_domain, div_weight)
		views_domain = views_domain**(1/3)

		## add first-view enhanced method
		# views_domain[1] *= 1.2
		# views_domain = np.divide(views_domain, np.sum(views_domain))

		pi_values = copy.deepcopy(views_domain)
		views_domain = views_domain*privacy_budget
		print('budget_allocation:', views_domain)

		laplace_scale = np.divide(laplace_scale, views_domain)
	print('laplace_scale:', laplace_scale,'\n')


	pi_value_set = {}
	idxx = 0
	for group in marginal_group:
		pi_value_set['/'.join(group)] = pi_values[idxx]
		idxx += 1
	store(pi_value_set, path_tmp + '/pi_value_set')
	del pi_value_set, idxx, pi_values
	

	marginal_table_dictionary = {}
	table_count = 0
	for group in tqdm(marginal_group, ncols = 90):
		marginal_table = {}
		marginal_name = '/'.join(group)
		attr_sizes = []
		for element in group:
			attr_sizes.append(range(attr_info[element]['max'] + 1))

		marginal_table_dictionary[marginal_name] = {}
		marginal_table_dictionary[marginal_name]['savePath'] = table_count

		attr_sizes_extend = list(itertools.product(*attr_sizes))
		del attr_sizes
		marginal_table = {ele:np.random.laplace(0, laplace_scale[table_count]) for ele in attr_sizes_extend}
		del attr_sizes_extend
		pickle_store(marginal_table, mypath + '/marginal_table_' + str(table_count))
		del marginal_table
		table_count += 1
	store(marginal_table_dictionary, path_tmp + '/marginal_table_dictionary')
	del marginal_table_dictionary, table_count, attr_info, marginal_group, mypath

def build_marginal(privacy_budget, temp_root, path_tmp, directory_name, attr_name):
    mypath = path_tmp + '/marginal_tables'
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    attr_info = load(temp_root + directory_name + '/' + directory_name +'_attr_info_max')
    laplace_scale = 1/privacy_budget

    attr_sizes = []
    for element in attr_name:
        attr_sizes.append(range(attr_info[element]['max'] + 1))
    attr_sizes_extend = list(itertools.product(*attr_sizes))

    marginal_table = {}
    marginal_table = {ele:np.random.laplace(0, laplace_scale) for ele in attr_sizes_extend}
    del attr_sizes_extend
    pickle_store(marginal_table, mypath + '/marginal_table')

    del mypath, marginal_table, laplace_scale,attr_info