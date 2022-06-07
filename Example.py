#Example code for running a FIBERS analysis and outputting a summary.
#Note: the code in the FIBERS_Algorithm_Code would have to be run first to define the necessary functions before running a FIBERS analysis

#Data inputted into FIBERS can only contain columns for (1) the features eligible for binning, (2) the variable indicating event occurrence, and (3) the variable
#indicating duration to event. Thus, preprocessing may be required prior to running FIBERS

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

#Loading in a dataset
data = pd.read_csv('Your Dataset Here")e

#Running the FIBERS algorithm
bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores, MAF_0_features = FIBERS (False, False, False, 500, data, 'grf_fail', 'graftyrs', 50, 2, 2, 0.2, 0.8, 0.4, 0.8)

#Printing a summary with risk stratification plots based on the top constructed bin
Top_Bin_Summary(grf_data, 'grf_fail', 'graftyrs', bin_feature_matrix, amino_acid_bins, amino_acid_bin_scores)
