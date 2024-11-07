import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
data = pd.read_csv('creditcard.csv',sep=',')
#print(data.head())
X = data.drop('Class', axis=1)  # Independent features
Y = data['Class']                # Dependent feature
state = np.random.RandomState(42)
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
#print(X.shape)
#print(Y.shape)
## Get the Fraud and the normal dataset 
fraud= data[data['Class']==1]
normal= data[data['Class']==0]
#print(fraud.shape, normal.shape)
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss

# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_resample(X,Y)
print(X_res.shape,y_res.shape)

## RandomOverSampler to handle imbalanced data
from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X, Y)
print(X_train_res.shape,y_train_res.shape)

