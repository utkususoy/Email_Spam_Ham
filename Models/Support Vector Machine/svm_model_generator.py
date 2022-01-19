from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score

np.random.seed(500)

def svm_holdout(X_train, X_test, y_train, y_test, c_, degree_, gamma_):
    
    svm_fit = svm.SVC(C=c_, kernel='linear', degree=degree_, gamma=gamma_)
    svm_model = svm_fit.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='Spam', average='binary')
    
    return svm_model, precision, recall, round((y_pred == y_test).sum() / len(y_pred), 3)

    
def svm_gridSearchCV(train_data, label, param, cv):
    
    gs = GridSearchCV(svm.SVC(), param, cv=5, n_jobs=-1)
    gs_fit = gs.fit(train_data, label)
    return gs_fit