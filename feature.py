# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 23:38:35 2016

@author: Leaf
"""

def get_faction(key1, key2, new_feature, data_dict): 
    """
    take in two feature key1 and key2, and create new_feature  to dictionary 
    with value equal to key1 value divided by key2 value """
    for i in data_dict:
        if (data_dict[i][key1] == 'NaN') or (data_dict[i][key2] == 'NaN'):
            data_dict[i][new_feature] = 0.0
        else:
            data_dict[i][new_feature] = float(data_dict[i][key1])/float(data_dict[i][key2])
                        

def get_total(key_list, new_feature, data_dict): 
    """
    take in a list of features, and create new_feature to dictionary 
    with value equal to sum of all featuers in the feature list """
    for i in data_dict:
        data_dict[i][new_feature] = 0.0
        for key in key_list:
            if data_dict[i][key] != 'NaN':
                data_dict[i][new_feature] =+ data_dict[i][key]            
            
            