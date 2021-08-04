# %%
import pandas as pd
import numpy as np

df = pd.read_csv("processed.cleveland.data.csv",header=0,sep=",")
df.dropna(inplace=True)

label = 'num'

# %%

df.head(5)

# %%
## convert categorical to one hot encoding
chest_pain = {1:'typical_angina',2:'atypical_angina',3:'non_anginal_pain',4:'asymptomatic'}
chest_pain_inv = {v:k for k, v in chest_pain.items()}
sex = {1:'male',0:'female'}
sex_inv = {v:k for k, v in sex.items()}
fbs_120 = {1:'fbs_true',0:'fbs_false'}
fbs_120_inv = {v:k for k,v in fbs_120.items()}
rest_ekg = {0:'ekg_normal',1:'ekg_having',2:'ekg_showing'}
exang = {1:'exang_yes',0:'exang_no'}
slope = {1:'upsloping',2:'flat',3:'downsloping'}
thal = {3:'thal_normal',6:'fixed_defect',7:'reversable_defect'}

# %%
## convert categorical numbers back to category so we
## can apply one hot encoding

df['cp']        = df['cp'].apply(lambda x: chest_pain.get(x))
df['fbs']       = df['fbs'].apply(lambda x: fbs_120.get(x))
df['restecg']   = df['restecg'].apply(lambda x: rest_ekg.get(x))
df['exang']     = df['exang'].apply(lambda x: exang.get(x))
df['slope']     = df['slope'].apply(lambda x: slope.get(x, "findme"))
df['thal']      = df['thal'].apply(lambda x: thal.get(x, "findme"))
df['sex']       = df['sex'].apply(lambda x: sex.get(x))



# %%
var_list = ['age','trestbps','chol','thalach','oldpeak']
for var in var_list:
    print(var.upper())
    print(df[var].describe())
    print("-------")
# %%
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')

# %%

df['nAge'] = qt.fit_transform(df[['age']].values.reshape(-1,1))
df['nTrestbps'] = qt.fit_transform(df[['trestbps']].values.reshape(-1,1))
df['nChol'] = qt.fit_transform(df[['chol']].values.reshape(-1,1))
df['nThalach'] = qt.fit_transform(df[['thalach']].values.reshape(-1,1))
df['nOldpeak'] = qt.fit_transform(df[['oldpeak']].values.reshape(-1,1))
df['nCa'] = qt.fit_transform(df[['ca']].values.reshape(-1,1))

# %%

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='whitegrid')

df_explore = df[['nAge','nTrestbps','nChol','nThalach','nOldpeak','nCa']]
for cat in df_explore:
    cat_num = df_explore[cat]
    print("Plot for %s: total count = %d" % (cat.upper(), len(cat_num)))
    f, ax = plt.subplots(figsize=(10,5))
    plot = sns.distplot(cat_num)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
    plt.show()


# %%

def init_encoder(df, col_names_list):
    d = {}
    for col_name in col_names_list:
        d[col_name] = df[col_name].unique().tolist()
    return d

def one_hot_encoder(df, var_dict):
    for var, vals in var_dict.items():
        for val in vals:
            df[val] = df[var].apply(lambda x: 1 if val in x else 0)
    return df

# %%

d = init_encoder(df, ['cp','fbs','restecg','exang','slope','exang','slope','sex'])
one_hot_encoder(df,d)

# %%

cleaned = df[['num', 'nAge', 'nTrestbps',
       'nChol', 'nThalach', 'nOldpeak', 'nCa', 'typical_angina', 'asymptomatic',
       'non_anginal_pain', 'atypical_angina', 'fbs_true', 'fbs_false',
       'ekg_showing', 'ekg_normal', 'ekg_having', 'exang_no', 'exang_yes',
       'downsloping', 'flat', 'upsloping', 'male', 'female']]
cleaned.to_csv("cleaned.csv", index=False, header=True)

# %%

label = 'num'
features = ['nAge','nTrestbps','nChol','nThalach','nOldpeak','nCa','typical_angina', 'asymptomatic',
       'non_anginal_pain', 'atypical_angina', 'fbs_true', 'fbs_false',
       'ekg_showing', 'ekg_normal', 'ekg_having', 'exang_no', 'exang_yes',
       'downsloping', 'flat', 'upsloping', 'male', 'female']


# %%

def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best
# %%

def stratified_split(df, target, val_percent=0.2):
    '''
    Function to split a dataframe into train and validation sets, while preserving the ratio of the labels in the target variable
    Inputs:
    - df, the dataframe
    - target, the target variable
    - val_percent, the percentage of validation samples, default 0.2
    Outputs:
    - train_idxs, the indices of the training dataset
    - val_idxs, the indices of the validation dataset
    '''
    classes=list(df[target].unique())
    train_idxs, val_idxs = [], []
    for c in classes:
        idx=list(df[df[target]==c].index)
        np.random.shuffle(idx)
        val_size=int(len(idx)*val_percent)
        val_idxs+=idx[:val_size]
        train_idxs+=idx[val_size:]
    return train_idxs, val_idxs


train_idxs, val_idxs = stratified_split(df, label, val_percent=0.25)

val_idxs, test_idxs = stratified_split(df[df.index.isin(val_idxs)], label, val_percent=0.5)

# %%

def test_stratified(df, col):
    '''
    Analyzes the ratio of different classes in a categorical variable within a dataframe
    Inputs:
    - dataframe
    - categorical column to be analyzed
    Returns: None
    '''
    classes=list(df[col].unique())
    
    for c in classes:
        print(f'Proportion of records with {c}: {len(df[df[col]==c])*1./len(df):0.2} ({len(df[df[col]==c])} / {len(df)})')

# %%
print('---------- STRATIFIED SAMPLING REPORT ----------')
print('-------- Label proportions in FULL data --------')
test_stratified(df, label)
print('-------- Label proportions in TRAIN data --------')
test_stratified(df[df.index.isin(train_idxs)], label)
print('------ Label proportions in VALIDATION data -----')
test_stratified(df[df.index.isin(val_idxs)], label)
print('-------- Label proportions in TEST data ---------')
test_stratified(df[df.index.isin(test_idxs)], label)

# %%
 
train_df = df[df.index.isin(train_idxs)]
X_train = train_df[features].values
Y_train = train_df[[label]].values
print('Retrieved Training Data')
val_df = df[df.index.isin(val_idxs)]
X_val = val_df[features].values
Y_val = val_df[[label]].values
print('Retrieved Validation Data')
test_df = df[df.index.isin(test_idxs)]
X_test = test_df[features].values
Y_test = test_df[[label]].values
print('Retrieved Test Data')        

# %%

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
#store data, all in numpy arrays
training_data = {'X_train':X_train,'Y_train':Y_train,
                'X_val': X_val,'Y_val':Y_val,
                'X_test': X_test,'Y_test':Y_test}

# %%

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=None,random_state=27,
                       verbose=1)
clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

# %%
predicted_labels = clf.predict(training_data['X_test'])
accuracy_score(training_data['Y_test'], predicted_labels)

# %%

params = {
    'n_estimators'      : range(100,500,50),
    'max_depth'         : [8, 9, 10, 11, 12],
    'max_features': ['auto'],
    'criterion' :['gini']
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch1 = GridSearchCV(estimator = clf, param_grid = params, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch1.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

# %%

GridSearchCV(cv=5, error_score=np.nan,
             estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=100, n_jobs=None,
                                              oob_score=False, random_state=27,
                                              verbose=1, warm_start=False),
             n_jobs=-1,
             param_grid={'criterion': ['gini'], 'max_depth': [8, 9, 10, 11, 12],
                         'max_features': ['auto'],
                         'n_estimators': range(100, 500, 50)},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='f1_micro', verbose=10)

# %%

getTrainScores(gsearch1)

# %%

clf2 = gsearch1.best_estimator_

params1 = {
    'n_estimators'      : range(200,300,10),
    'max_depth'         : [11, 12,13]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch2 = GridSearchCV(estimator = clf2, param_grid = params1, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch2.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))

# %%
GridSearchCV(cv=5, error_score=np.nan,
             estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=12,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=250, n_jobs=None,
                                              oob_score=False, random_state=27,
                                              verbose=1, warm_start=False),
             n_jobs=-1,
             param_grid={'max_depth': [11, 12, 13],
                         'n_estimators': range(200, 300, 10)},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='f1_micro', verbose=10)
# %%
getTrainScores(gsearch2)
# %%
clf3 = gsearch2.best_estimator_

params2 = {
    'n_estimators'      : range(200,220,5),
    'max_depth'         : [13,14,15]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch3 = GridSearchCV(estimator = clf3, param_grid = params2, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch3.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
# %%
GridSearchCV(cv=5, error_score=np.nan,
             estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=13,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=210, n_jobs=None,
                                              oob_score=False, random_state=27,
                                              verbose=1, warm_start=False),
             n_jobs=-1,
             param_grid={'max_depth': [13, 14, 15],
                         'n_estimators': range(200, 220, 5)},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='f1_micro', verbose=10)
# %%
getTrainScores(gsearch3)
# %%# %%
clf4 = gsearch3.best_estimator_

params3 = {
    'max_depth'         : range(14,20,1)
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch4 = GridSearchCV(estimator = clf4, param_grid = params3, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch4.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
# %%# %%
GridSearchCV(cv=5, error_score=np.nan,
             estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=15,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=210, n_jobs=None,
                                              oob_score=False, random_state=27,
                                              verbose=1, warm_start=False),
             n_jobs=-1,
             param_grid={'max_depth': range(14, 20)}, pre_dispatch='2*n_jobs',
             refit=True, return_train_score=False, scoring='f1_micro',
             verbose=10)
# %%# %%
clf5 = gsearch4.best_estimator_

params4 = {
    'max_depth'         : range(19,50,2)
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch5 = GridSearchCV(estimator = clf5, param_grid = params4, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch5.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
# %%# %%
GridSearchCV(cv=5, error_score=np.nan,
             estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=19,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=210, n_jobs=None,
                                              oob_score=False, random_state=27,
                                              verbose=1, warm_start=False),
             n_jobs=-1,
             param_grid={'max_depth': range(19, 50, 2)},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='f1_micro', verbose=10)
# %%

getTrainScores(gsearch5)

# %%
clf6 = gsearch5.best_estimator_

params5 = {
    'max_depth'         : [24,25,26]
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch6 = GridSearchCV(estimator = clf6, param_grid = params5, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch6.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
# %%
GridSearchCV(cv=5, error_score=np.nan,
             estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                              class_weight=None,
                                              criterion='gini', max_depth=25,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              max_samples=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators=210, n_jobs=None,
                                              oob_score=False, random_state=27,
                                              verbose=1, warm_start=False),
             n_jobs=-1,
             param_grid={'max_depth': [24, 25, 26]}, pre_dispatch='2*n_jobs',
             refit=True, return_train_score=False, scoring='f1_micro',
             verbose=10)
# %%
getTrainScores(gsearch6)
# %%
final_clf = gsearch6.best_estimator_
# %%
final_clf.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
predicted_labels = final_clf.predict(training_data['X_test'])
train_pred = final_clf.predict(training_data['X_train'])
print('Train Accuracy:'+str(accuracy_score(training_data['Y_train'], train_pred)))
print('Train F1-Score(Micro):'+str(f1_score(training_data['Y_train'], train_pred,average='micro')))
print('------')
print('Test Accuracy:'+str(accuracy_score(training_data['Y_test'], predicted_labels)))
print('Test F1-Score(Micro):'+str(f1_score(training_data['Y_test'], predicted_labels,average='micro')))
# %%
f, ax = plt.subplots(figsize=(10,5))
plot = sns.barplot(x=features, y=final_clf.feature_importances_)
ax.set_title('Feature Importance')
plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
plt.show()

# %% [markdown]
## XGBoost

# %%

import xgboost as xgb
import matplotlib.pyplot as plt

#allow logloss and classification error plots for each iteration of xgb model
def plot_compare(metrics,eval_results,epochs):
    for m in metrics:
        test_score = eval_results['val'][m]
        train_score = eval_results['train'][m]
        rang = range(0, epochs)
        plt.rcParams["figure.figsize"] = [6,6]
        plt.plot(rang, test_score,"c", label="Val")
        plt.plot(rang, train_score,"orange", label="Train")
        title_name = m + " plot"
        plt.title(title_name)
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.show()
        
def fitXgb(sk_model, training_data=training_data,epochs=300):
    print('Fitting model...')
    sk_model.fit(training_data['X_train'], training_data['Y_train'].reshape(training_data['Y_train'].shape[0],))
    print('Fitting done!')
    train = xgb.DMatrix(training_data['X_train'], label=training_data['Y_train'])
    val = xgb.DMatrix(training_data['X_val'], label=training_data['Y_val'])
    params = sk_model.get_xgb_params()
    metrics = ['mlogloss','merror']
    params['eval_metric'] = metrics
    store = {}
    evallist = [(val, 'val'),(train,'train')]
    xgb_model = xgb.train(params, train, epochs, evallist,evals_result=store,verbose_eval=100)
    print('-- Model Report --')
    print('XGBoost Accuracy: '+str(accuracy_score(sk_model.predict(training_data['X_test']), training_data['Y_test'])))
    print('XGBoost F1-Score (Micro): '+str(f1_score(sk_model.predict(training_data['X_test']),training_data['Y_test'],average='micro')))
    plot_compare(metrics,store,epochs)
    features = ['nAge', 'nTrestbps', 'nChol', 'nThalach', 'nOldpeak', 'nCa', 'typical_angina', 'asymptomatic', 'non_anginal_pain', 'atypical_angina', 'fbs_true', 'fbs_false', 'ekg_showing', 'ekg_normal', 'ekg_having', 'exang_no', 'exang_yes', 'downsloping', 'flat', 'upsloping', 'male', 'female']
    f, ax = plt.subplots(figsize=(10,5))
    plot = sns.barplot(x=features, y=sk_model.feature_importances_)
    ax.set_title('Feature Importance')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.show()

# %% [markdown]
## XGBoost hyperparameter tuning using GridsearchCV

# %%
## get the size of the class label
num_class = df[label].unique().size
num_class

# %%

from xgboost.sklearn import XGBClassifier
#initial model
xgb1 = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=num_class,
                    seed=27)

# %%
fitXgb(xgb1, training_data)

# %%

def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best

# %%

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
#metrics to consider: f1_micro, f1_macro, roc_auc_ovr
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=5)
gsearch1.fit(X_train, Y_train)

# %%

GridSearchCV(cv=5, error_score=np.nan,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=0.8, gamma=0, gpu_id=None,
                                     importance_type='gain',
                                     interaction_constraints=None,
                                     learning_rate=0.1, max_delta_step=None,
                                     max_depth=5, min_child_weight=1,
                                     missing=np.nan, monotone_constraints=None,
                                     n_estimators=1000,
                                     objective='multi:softmax',
                                     random_state=None, reg_alpha=None,
                                     reg_lambda=None, scale_pos_weight=None,
                                     seed=27, subsample=0.8, tree_method=None,
                                     validate_parameters=None, verbosity=None),
             n_jobs=-1,
             param_grid={'max_depth': range(3, 10, 2),
                         'min_child_weight': range(1, 6, 2)},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='f1_micro', verbose=10)

# %%
getTrainScores(gsearch1)

# %%
xgb2 = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=3,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=num_class,
                    seed=27)

fitXgb(xgb2, training_data)

# %%
param_test2 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch2 = GridSearchCV(estimator = xgb2, param_grid = param_test2, scoring='f1_micro',n_jobs=-1,verbose = 10, cv=3)
gsearch2.fit(X_train, Y_train)

# %%
GridSearchCV(cv=3, error_score=np.nan,
             estimator=XGBClassifier(base_score=0.5, booster='gbtree',
                                     colsample_bylevel=1, colsample_bynode=1,
                                     colsample_bytree=0.8, gamma=0, gpu_id=-1,
                                     importance_type='gain',
                                     interaction_constraints='',
                                     learning_rate=0.1, max_delta_step=0,
                                     max_depth=5, min_child_weight=3,
                                     missing=np.nan, monotone_constraints='()',
                                     n_estimators=1000, n_jobs=4,
                                     num_class=num_class, num_parallel_tree=1,
                                     objective='multi:softprob',
                                     random_state=27, reg_alpha=0, reg_lambda=1,
                                     scale_pos_weight=None, seed=27,
                                     subsample=0.8, tree_method='exact',
                                     validate_parameters=1, verbosity=None),
             n_jobs=-1,
             param_grid={'reg_alpha': [1e-05, 0.01, 0.1, 1, 100]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='f1_micro', verbose=10)

# %%
getTrainScores(gsearch2)

# %%
xgb3 = XGBClassifier(learning_rate=0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=4,
                    num_class=9,
                    seed=27)

fitXgb(xgb3, training_data)

# %% [markdown]
### Model Deploy

# %%
import pickle

pickl = {'model': xgb3}
pickle.dump(pickl, open('model_file'+'.p','wb'))

# %%

file_name = 'model_file.p'
with open(file_name,'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

# %%
#input transformed values to make a prediction in FlaskAPI
model.predict(X_test[4,:].reshape(1,-1))

# %%


# %%

# %%


# %%
