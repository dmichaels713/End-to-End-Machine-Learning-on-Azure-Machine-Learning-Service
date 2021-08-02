# Load in libaries
import json
import numpy as np
import os
import pandas as pd
from itertools import zip_longest

# Creates a Python Dictionary object out of key-value pairs
def create_dict(keys, values):
    return dict(zip_longest(keys, values[:len(keys)]))

# Sets tags for Azure resources
def set_tags(tagNameList, tagValueList):
    return create_dict(tagNameList, tagValueList)

# Writes CSV files back to a storage account
def write_to_datastore(dataframe, workspace, datastore, folder, file, indexBoolean):
    os.makedirs(folder, exist_ok=True) 
    filePath = os.path.join(folder, file)
    dataframe.to_csv(filePath, index = indexBoolean)
    print('Data Written as CSV and saved to ' + filePath)
    
    # Upload to Datastore
    files = [filePath]
    datastore.upload_files(files=files, target_path=folder, overwrite = True)
    
# Split data into X (features) and Y (target) columns for machine learning
def split_x_y (dataframe, scoring_column):
    X = dataframe.drop(scoring_column, axis=1)
    Y = dataframe[scoring_column]
    return X, Y
    
# Make predictions using a classification model
def make_classification_predictions(model, dataframe, X, Y):
    dataframe.loc[:, 'Predictions'] = model.predict(X)
    for i in range(0, len(Y.unique())):
        dataframe.loc[:, 'Probability_' + str(Y.unique()[i])] = model.predict_proba(X)[:,i]
    return dataframe

# Saves local explanations as columns to a dataframe
def save_local_explanations(explainerModel, dataframe, features):
    localExplanation = explainerModel.explain_local(features)
    localImportanceNames = localExplanation.get_ranked_local_names()
    localImportanceValues = localExplanation.get_ranked_local_values()
    dataframe['ExplanationColumns'] = localImportanceNames[0]
    dataframe['ExplanationValues'] = localImportanceValues[0]
    dataframe['Explanations'] = 'fill'
    for i in range(0,len(dataframe)):
        dataframe['Explanations'][i] = json.dumps(dict(zip(dataframe.ExplanationColumns[i],\
                                                           dataframe.ExplanationValues[i])))
    dataframe = dataframe.drop(['ExplanationColumns','ExplanationValues'], axis=1)
    return dataframe

# Saves local explanations as columns to a dataframe
def save_global_explanations(explainerModel, features):
    globalExplanation = explainerModel.explain_global(features)
    pd.DataFrame()
    global_names = globalExplanation.get_ranked_global_names()
    global_values = globalExplanation.get_ranked_global_values()
    global_zipped = list(zip(global_names, global_values))
    dataframe = pd.DataFrame(global_zipped, columns=['Columns','Importance'])
    return dataframe
