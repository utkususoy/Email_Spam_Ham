from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score

np.random.seed(500)

def random_forest_holdout(X_train, X_test, y_train, y_test, n_estimators_, max_depth_):
    
    rf = RandomForestClassifier(n_estimators=n_estimators_, max_depth=max_depth_, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='Spam', average='binary')
    
    return rf_model, precision, recall, round((y_pred == y_test).sum() / len(y_pred), 3)

    
def rf_gridSearchCV(train_data, label, param, cv):
    rf = RandomForestClassifier()
    gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)
    gs_fit = gs.fit(train_data, label)
    return gs_fit