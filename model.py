# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:44:30 2016

@author: Leaf
"""
import sys
sys.path.append("../tools/")
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import pandas as pd

# create classifier step for pipeline
def create_classifier_step(algorithm):
    cl_params = {}
    # to avoid unbound local error, assign cl a default value
    cl = GaussianNB()
    if algorithm == 'Naive_Bayes':
        cl = GaussianNB()
    
    elif algorithm == 'Logistic_Regression':
        cl = LogisticRegression()
        cl_params = {algorithm + '__C' : [10, 100, 1000, 10000]}       
        
    elif algorithm == 'SVM':
        cl = SVC()
        cl_params = {algorithm + '__kernel' : ['linear', 'rbf', 'poly'],
                     algorithm + '__C' : [10, 100, 1000, 10000]}
                     
    elif algorithm == 'Decision_Tree':
        cl = DecisionTreeClassifier()
        cl_params = {algorithm + '__min_samples_split' : [30, 40, 50, 60]}
        
    elif algorithm == 'KNN':
        cl = KNeighborsClassifier()
        cl_params = {algorithm + '__n_neighbors' : [4, 6, 8, 10],
                     algorithm + '__weights' : ['uniform', 'distance']}
        
    elif algorithm == 'Random_Forest':
        cl = RandomForestClassifier()
        cl_params = { algorithm + '__n_estimators' : [2, 5, 7, 10, 12],
                      algorithm + '__max_features' : ['sqrt', 'log2']}
    
    return (algorithm, cl), cl_params
    
""" 
create pipeline which:
    1. Standardize the features to be centered around 0 with a standard
    deviation of 1 by using StandardScaler
    2. Select features using SelectKBest and Anova F-value classification scoring 
    3. Reduce dimensionality using Principal Component Analysis
    4. Feed the resulting PCA components to classification algorithms.
"""

def create_pipeline(algorithm, features_train, labels_train, feature_list):

    clf_step, clf_params = create_classifier_step(algorithm)

    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                           ('select', SelectKBest()),
                           #('PCA', PCA()),
                           clf_step])
    
    params = {#'PCA__n_components': [2, 3, 4],
              'select__k' : [6, 7, 8, 9, 10, 11, 12, 13],
              'select__score_func' : [f_classif]}
              
    params.update(clf_params)

    sss = StratifiedShuffleSplit(labels_train, n_iter = 100, 
                                 test_size = 0.3, random_state = 42)
                                 
    gscv = GridSearchCV(pipe, params, verbose = 0, 
                        scoring = 'recall', cv=sss)
    gscv.fit(features_train, labels_train)
    
    clf = gscv.best_estimator_
        
    selected_features = clf.named_steps['select'].get_support(indices=True)
    
    feature_scores = clf.named_steps['select'].scores_
    sfs = pd.DataFrame([feature_list[i+1], feature_scores[i]] for i in selected_features)
    sfs.columns = ['Features', 'Scores']
    sfs = sfs.sort_values(by=['Scores'], ascending=False).reset_index(drop=True)
    
    return clf, sfs

    