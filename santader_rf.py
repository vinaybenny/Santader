# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:38:09 2016

@author: vinay.benny
"""

import numpy as  np;
import pandas as pd;
import os;
from sklearn.feature_selection import VarianceThreshold;
import itertools;
from sklearn.preprocessing import StandardScaler;
import xgboost as xgb;
from sklearn.cross_validation import train_test_split;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.grid_search import GridSearchCV;


def rem_constant_feat(pd_dataframe):
    selector = VarianceThreshold();
    selector.fit(pd_dataframe);
    variant_indices = selector.get_support(indices=True);
    all_indices = np.arange(pd_dataframe.columns.size);
    nonvariant_indices = np.delete(all_indices, variant_indices);
    pd_dataframe = pd_dataframe.drop(labels=pd_dataframe[nonvariant_indices], axis=1);
    print("  - Deleted %s / %s features (~= %.1f %%)" % (nonvariant_indices.size, 
          all_indices.size ,100.0 * (np.float(nonvariant_indices.size) / all_indices.size)));
    return pd_dataframe, nonvariant_indices;
    
def rem_identical_feat(pd_dataframe):
    delete_feats = [];
    n_features = pd_dataframe.shape[1]
    for feat1, feat2 in itertools.combinations(iterable=pd_dataframe.columns, r=2):
        if np.array_equal(pd_dataframe[feat1], pd_dataframe[feat2]):
            delete_feats.append(feat2);
        
    delete_feats = np.unique(delete_feats);
    pd_dataframe = pd_dataframe.drop(labels=delete_feats, axis=1);
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
    len(delete_feats), n_features,
    100.0 * (np.float(len(delete_feats)) / n_features)))
    return pd_dataframe, delete_feats;
    
def apply_norm_feat(pd_dataframe):    
    normalizer = StandardScaler();    
    norm_xtrain = normalizer.fit_transform(xtrain); 
    print("  - Normalized the training dataset");
    return norm_xtrain, normalizer;


if __name__ == "__main__":    
    np.random.seed(93);    
    os.chdir("C:/Users/vinay.benny/Documents/Kaggle/Santader");
    train = pd.DataFrame.from_csv("train.csv");
    test = pd.DataFrame.from_csv("test.csv");
    
    #Remove constant features in train, and drop the same in test    
    train, indices = rem_constant_feat(train);
    test = test.drop(labels=test[indices], axis=1);
    
    #Remove identical features in training datasets, and drop the same in test
    train, feats = rem_identical_feat(train);
    test = test.drop(labels=feats, axis=1);    
       
    # Apply normalization to training data, and use the normalizer on test
    ytrain = train["TARGET"];
    xtrain = train.drop(labels="TARGET", axis=1);        
    norm_xtrain, normalizer = apply_norm_feat(xtrain);
    norm_xtest = normalizer.transform(test);   
    
    # Train a random forest classifier    
    rf = RandomForestClassifier(n_estimators=20);
    param_grid_vals={'max_depth': [5,7,10], "criterion": ["gini", "entropy"], 'oob_score' : [True]}
    estimator = GridSearchCV(rf, param_grid=param_grid_vals, verbose=2); 
    estimator.fit(norm_xtrain, ytrain);
    
    y_test = estimator.best_estimator_.predict_proba(norm_xtest)
    
    
    submission = pd.DataFrame({"ID": test.index.values.astype(str), "TARGET": y_test[:,1]});
    submission.to_csv("rf_submission1.csv", index=False);
    
    
    
    
    
    
    
    
       