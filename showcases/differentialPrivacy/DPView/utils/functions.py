import pickle
import json
import math


def store(data, save_path):
    with open(save_path + '.json', 'w') as fw:
        json.dump(data,fw)
        
def load(load_path):
    with open(load_path + '.json','r') as f:
        data = json.load(f)
        return data

def pickle_store(data, save_path):
	with open(save_path + '.pickle', 'wb') as f:
		pickle.dump(data, f)

def pickle_load(load_path):
	with open(load_path + '.pickle', 'rb') as f:
		data = pickle.load(f)
		return data
    
def ceil(x):
    return math.ceil(x)
	
def rename_df(dict_name, list_name):
	output = {}
	list_idx = 0
	for element in dict_name[:-1]:
		output[element] = list_name[list_idx]
		list_idx += 1
	return output