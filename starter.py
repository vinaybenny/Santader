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
from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from matplotlib import pyplot as plt;
from sklearn.pipeline import Pipeline;
from sklearn.linear_model import LogisticRegression;
from sklearn.grid_search import GridSearchCV;
from sklearn.cross_validation import cross_val_score;


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

def apply_PCA(xtrain):    
       
    pca = PCA(n_components=100);
    #logistic = LogisticRegression();
    #pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)]); 
    
    #n_components = [100, 150, 200];
    #Cs=np.logspace(-4, 4, 3);
    #estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs), verbose=2);
    #estimator.fit(xtrain, ytrain);
    xtrain_comps = pca.fit_transform(xtrain);
    print("  - Applied PCA on the training dataset");
    return xtrain_comps, pca;
    
def pca_visualize(pca, xtrain, ytrain):
    
    classes = np.sort(np.unique(ytrain))
    labels = ["Satisfied customer", "Unsatisfied customer"]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    colors = [(0.0, 0.63, 0.69), 'black']
    markers = ["o", "D"]
    for class_ix, marker, color, label in zip(
            classes, markers, colors, labels):
        ax.scatter(xtrain[np.where(ytrain == class_ix), 0],
                   xtrain[np.where(ytrain == class_ix), 1],
                   marker=marker, color=color, edgecolor='whitesmoke',
                   linewidth='1', alpha=0.9, label=label)
        ax.legend(loc='best')
    plt.title(
        "Scatter plot of the training data examples projected on the "
        "2 first principal components")
    plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (
        pca.explained_variance_ratio_[0] * 100.0))
    plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (
        pca.explained_variance_ratio_[1] * 100.0))
    plt.show();
    return;
    
    
def make_submission(clf, xtest, ids, name='submission_test.csv'):
    y_prob = clf.predict_proba(xtest)
    with open(name, 'w') as f:
        f.write('ID')
        f.write(',TARGET')
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))    

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
    
    #Apply principal components analysis on training, and transform the test data 
    #xtrain_comps, pcaest = apply_PCA(norm_xtrain);
    pca = PCA();
    #xtest_comps = pcaest.transform(norm_xtest);
    #pca_visualize(pcaest, xtrain_comps, ytrain);
    
    #Apply logistic regression on training data
    logistic = LogisticRegression(); 
    #param_grid_vals = {'C':np.logspace(-4, 4, 3)}
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)]);
    #estimator = GridSearchCV(logistic, param_grid=param_grid_vals, verbose=2);
    n_components = [100];
    Cs=np.logspace(-5, -1, 5);
    estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs), verbose=2);
    estimator.fit(norm_xtrain, ytrain);
    
    ytest_prob = make_submission(estimator, norm_xtest, test.index.values.astype(str));
    
    
    
    
    
    
    
    
       