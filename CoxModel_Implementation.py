#Example of running a covariate-adjusted Cox model to evaluate the top bin constructed by FIBERS
#This code would be run after defining the FIBERS algorithm and subfunctions with code in FIBERS_Algorithm_Code.py and then running a FIBERS analysis (see Example.py)

#Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from random import choice
from random import seed
from random import randrange
import collections
import statistics
import math
import numpy as numpy
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

#Accessing the top bin
sorted_bin_scores = dict(sorted(amino_acid_bin_scores.items(), key=lambda item: item[1], reverse=True))
sorted_bin_list = list(sorted_bin_scores.keys())
sorted_bin_feature_importance_values = list(sorted_bin_scores.values())
    
Bin = amino_acid_bins[sorted_bin_list[0]]

#"data" is a feature matrix including all covariates, features eligible for binning, and occurrence/duration variabels
d_data = data.copy()
d_data['Bin'] = d_data[Bin].sum(axis=1)
column_values = d_data['Bin'].to_list()
for r in range(0, len(column_values)):
    if column_values[r] > 0:
        column_values[r] = 1
data['Bin'] = column_values

#list of  covariates presented here (covariates, Bin variable, and occurrence/duration variables go into the Cox model)
coxmodeldata = data[['graftyrs', 'grf_fail', 'DON_AGE', 'REC_AGE_AT_TX', 'yearslice', 'diab_noted', 'DCD', 'ln_don_wgt_kg_0c', 'ln_don_wgt_kg_0c_s55', 'dcadcodanox', 'dcadcodcva', 'dcadcodcnst', 'dcadcodoth', 'don_ecd', 'don_htn_0c', 'mmA0', 'mmA1', 'mmB0', 'mmB1', 'mmDR0', 'mmDR1', 'mm0', 'mmC0', 'mmC1', 'mmDQ0', 'mmDQ1', 'shared', 'PKPRA_1080', 'PKPRA_GE80', 'PKPRA_MS', 'don_cmv_negative', 'rbmi_miss', 'rbmi_gt_20', 'can_dgn_htn_ndm', 'can_dgn_pk_ndm', 'can_dgn_gd_ndm', 'rec_age_spline_35', 'rec_age_spline_50', 'rec_age_spline_65', 'rbmi_DM', 'rbmi_gt_20_DM', 'dm_can_age_spline_50', 'ln_c_hd_0c', 'ln_c_hd_m', 'rec_prev_ki_tx', 'rec_prev_ki_tx_dm', 'age_diab', 'age_ecd', 'CAN_RACE_WHITE', 'hispanic', 'CAN_RACE_BLACK', 'CAN_RACE_asian', 'Bin']]
cat_columns = coxmodeldata.select_dtypes(['object']).columns
coxmodeldata[cat_columns] = coxmodeldata[cat_columns].apply(lambda x: pd.factorize(x)[0])


#Running and printing the Cox model results
cph = CoxPHFitter()
cph.fit(coxmodeldata,"graftyrs",event_col="grf_fail", show_progress=True)
cph.print_summary()
