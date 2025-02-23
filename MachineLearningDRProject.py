from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#Set seed for reproducability 
random_seed = 2
np.random.seed(random_seed)
random.seed(random_seed)

#read in data
diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)

X = diabetic_retinopathy_debrecen.data.features
y = diabetic_retinopathy_debrecen.data.targets
z = diabetic_retinopathy_debrecen.data

#Split into train, development,  and test datasets
X_train = X.sample(frac = 0.7, random_state=random_seed)
X_mid= X.drop(X_train.index)
X_dev = X_mid.sample(frac = 0.5,random_state=random_seed)
X_test = X_mid.drop(X_dev.index)

y_train = y.sample(frac = 0.7, random_state=random_seed)
y_mid= y.drop(y_train.index)
y_dev = y_mid.sample(frac = 0.5,random_state=random_seed)
y_test = y_mid.drop(X_dev.index)

#Standardize data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_dev_std = sc.transform(X_test)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

print('Weighted Accuracy: %.2f' % balanced_accuracy_score(y_test, y_pred))

print('F1 Score: %.2f' % f1_score(y_test, y_pred))