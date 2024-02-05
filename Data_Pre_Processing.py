import pandas as pd
import numpy as np
#import sklearn as sl
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.datasets import make_classification
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler,  normalize
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, SequentialFeatureSelector

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN 

def data_pre_processing_c():
    pass