#!/usr/bin/python

import os
os.chdir('/Users/Leaf/Documents/Nanodegree/P5/ud120-projects/final_project')
import sys
sys.path.append("../tools/")
import pickle
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.cross_validation import train_test_split
import feature
import model

##############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
  
print 'Total number of data points: %d' % len(data_dict)

poi_count = 0
for point in data_dict.keys():
    poi_count += data_dict[point]['poi']    
print 'Number of Persons of Interest: %d' % poi_count  
    
all_features = data_dict['BUY RICHARD B'].keys()
print 'Features for each person:  %s' %  all_features
 

############################################################################## 
### Task 2: Remove outliers

# check for outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

salary = []
bonus = []
for i in range(len(data)):
    salary.append(data[i][0])
    bonus.append(data[i][1]) 
       
plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

# find the point with maxium salary
for key in data_dict.keys():
    if (data_dict[key]['salary'] == max(salary)):
        print key, data_dict[key]
        
# drop this outlier        
data_dict.pop('TOTAL', 0)

data_dict['LOCKHART EUGENE E']
data_dict.pop('LOCKHART EUGENE E', 0)


features = ["salary", "bonus"]
data = featureFormat(data_dict, features)     
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus )

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()


############################################################################## 
### Task 3: Create new feature(s)

feature.get_faction("from_poi_to_this_person","to_messages", "fraction_from_poi_email",data_dict)
feature.get_faction("from_this_person_to_poi","from_messages", "fraction_to_poi_email",data_dict)

features = ["poi","fraction_from_poi_email", "fraction_to_poi_email"]
data = featureFormat(data_dict, features)     
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color='red')
    else:
        plt.scatter(from_poi, to_poi)

plt.xlabel("Email Fraction From POI")
plt.ylabel("Email Fraction To POI")
plt.show()


feature.get_total(["from_poi_to_this_person", "from_poi_to_this_person",
                   "shared_receipt_with_poi"], "total_poi_interaction", data_dict)
feature.get_total(['total_payments', 'total_stock_value'], 
                  'total_compensation', data_dict)

# create pandas Dataframe from dataset
full_df = pd.DataFrame.from_dict(data_dict, orient='index') 

# plot the new features
ax = sns.boxplot(x="poi", y="total_poi_interaction", data=full_df)
ax = sns.swarmplot(x="poi", y="total_poi_interaction", data=full_df, color=".25")

ax = sns.boxplot(x="poi", y="total_compensation", data=full_df)
ax = sns.swarmplot(x="poi", y="total_compensation", data=full_df, color=".25")

#count number of NaN for each column
full_df.replace('NaN', np.nan, inplace=True)
full_df.isnull().sum()
              
# imputate missing value with 0
full_df = full_df.replace(np.nan, 0)

# remove unuseful column
df = full_df.drop('email_address', axis = 1)

# correlation of columns with 'poi'
corrs = df.corr()
corrs.sort_values(by = ['poi'], ascending=False)['poi']

# drop label column and get features dataframe
features_df = full_df.drop(['email_address', 'poi'], axis = 1).astype(float)

# examine the importance of features
selection = SelectKBest(k='all', score_func=f_classif).fit(features_df, df['poi'])    
scores = pd.DataFrame([features_df.columns, selection.scores_]).T
scores.columns = ['Features', 'Scores']
scores = scores.sort_values(by=['Scores'], ascending=False).reset_index(drop=True)
print scores


##############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# use the first 20 features by SelectKBest
features_list = np.hstack((['poi'], scores['Features'][:20].tolist()))

# extract features and labels from dataset for local testing
# transform dataframe to dictionary
my_dataset = df.T.to_dict()
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
##############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

algorithms = ['Naive_Bayes', 'Logistic_Regression', 'SVM', 'Decision_Tree', 'KNN', 'Random_Forest']

def run_algorithm(algorithm):    
    print '\nRunning', algorithm, '----------'
    clf, selected_features = model.create_pipeline(algorithm, features_train, labels_train, features_list)
    print 'Selected features with scores:'
    print selected_features, '\n'
    print test_classifier(clf, my_dataset, features_list), '\n'

for algo in algorithms:
    run_algorithm(algo)    


clf = 'Naive_Bayes'

##############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)