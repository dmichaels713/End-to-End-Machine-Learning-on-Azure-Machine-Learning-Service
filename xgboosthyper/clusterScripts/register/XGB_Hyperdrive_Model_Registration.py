# Load in libraries
import argparse
import datetime as dt
import joblib
import json
import os
import pandas as pd
import pytz
import scipy.stats as st
import shutil
from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel
from itertools import zip_longest
from lightgbm import LGBMClassifier
from shutil import copy2
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Load in Azure libraries
from azureml.core import Dataset, Datastore, Experiment, Model, Run, Workspace

# change the directory so the helper functions can be found

# Load in functions from shared functions file
from util.XGB_Hyperdrive_Shared_Functions import create_dict, set_tags, save_local_explanations, write_to_datastore
from util.XGB_Hyperdrive_Shared_Functions import make_classification_predictions, split_x_y, save_global_explanations

# Define script-specific functions
def register_model(workspace, modelName, modelPath, trainDataset, valDataset, description, tags):
    Model.register(workspace = workspace, model_name = modelName, model_path = modelPath, description = description,\
                   tags = tags, datasets=[('Training', trainDataset),('Validation', valDataset)])
    print("Registered version {0} of model {1}".format(model.version, model.name))

def load_model_from_hd(modelFolder, savedModel):
    copy2(savedModel, modelFolder)
    modelPath = modelFolder + 'saved_model'
    model = joblib.load(modelPath)
    return model

def load_explainer_model_from_hd(modelFolder, explainerModel):
    copy2(explainerModel, modelFolder)
    modelPath = modelFolder + 'explainer_model'
    model = joblib.load(modelPath)
    return model

def load_model(modelName):
    modelPath = Model.get_model_path(modelName)
    model = joblib.load(modelPath)
    return model

def score_model(model, features, targetColumn, scoringMethod):
    predictions = model.predict(features)
    score = scoringMethod(targetColumn, predictions)
    return score

def set_scoring_method(scoringMethod):
    if scoringMethod == 'Accuracy Training':
        return accuracy_score
    elif scoringMethod == 'Balanced Accuracy Training':
        return balanced_accuracy_score
    elif scoringMethod == 'Precision Training':
        return precision_score
    elif scoringMethod == 'Recall Training':
        return recall_score
    elif scoringMethod == 'F1 Training':
        return F1_score
    else:
        print('Add your scoring metric to the set_scoring_method function')
        raise Exception ('Scoring Metric not found in set_scoring_method function.  Add your metric to the function.')

def init():
    # Set Arguments.
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str,
                            help='Name of Training Dataset')
    parser.add_argument('--val_dataset_name', type=str,
                            help='Name of Validation Dataset')
    parser.add_argument('--datastore_name', type=str,
                            help='Name of Datastore')
    parser.add_argument('--project_name', type=str,
                            help='Name of project')
    parser.add_argument('--project_description', type=str,
                            help='Description of project')
    parser.add_argument('--pytz_time_zone', type=str,
                            help='Time Zone associated with your data')
    parser.add_argument('--target_column_name', type=str,
                            help='Name of variable to score')
    parser.add_argument('--k_folds', type=int, default = 10,
                            help='Number of folds to split your data into for cross validation')
    parser.add_argument('--confidence_level', type=float, default = 0.95,
                            help='Level of confidence to set for your confidence interval ()')
    parser.add_argument('--model_name', type=str,
                            help='Name of model to register')
    parser.add_argument('--output_path', type=str,
                            help='Location to store output on Datastore')
    parser.add_argument('--scoring_metric', type=str,
                            help='Metric with which you scored your Hyperdrive Run')
    parser.add_argument('--metric_goal', type=str, default = 'MAXIMIZE',
                            help='Whether the scoring metric should be minimized or maximized')
    parser.add_argument('--saved_model', type=str, 
                            help='path to saved model file')
    parser.add_argument('--explainer_model', type=str, 
                            help='path to saved explanation file')
    parser.add_argument('--metrics_data', type=str,
                            help='Location of Hyperdrive Run Metrics File')
    args = parser.parse_args()

    # Set the Run context for logging
    global run
    run = Run.get_context()

def main():
    # Set scoring metric
    scoringMethod = set_scoring_method(args.scoring_metric)
    print ('Scoring Metric Set')
    
    # Retrieve your Metrics Data file
    with open(args.metrics_data) as metrics:
        metricsData = json.load(metrics)
    print('Metrics File Downloaded')
    
    # Turn the Metrics JSON file into a Pandas Dataframe, then transpose and sort it by your scoring metric.
    metrics = pd.DataFrame(metricsData)
    metricsTransposed = metrics.transpose().sort_values(by=args.scoring_metric, ascending=False)
    print('Metrics Dataframe Created')
    
    # Connect to your AMLS Workspace and set your Datastore
    ws = run.experiment.workspace
    datastore = Datastore.get(ws, args.datastore_name)
    print('Datastore Set')
    
    # Retrieve your dataset
    trainDataset = Dataset.get_by_name(ws, args.train_dataset_name)
    valDataset = Dataset.get_by_name(ws, args.val_dataset_name)
    print('Datasets Retrieved')
    
    # Transform your data into Pandas dataframes
    trainDF = trainDataset.to_pandas_dataframe()
    valDF = valDataset.to_pandas_dataframe()
    print('Datasets Converted to Pandas')
    
    # Split out X and Y variables
    val_X, val_Y = split_x_y(valDF, args.target_column_name)
    print("Validation Data split into Feature and Target Columns")
    
    # Load your training model
    modelFolder = 'model/'
    os.makedirs(modelFolder, exist_ok=True)
    newModel = load_model_from_hd(modelFolder, args.saved_model)
    print("Training Model Loaded")
    
    # Load your explainer model
    explainer = load_explainer_model_from_hd(modelFolder, args.explainer_model)
    print("Explainer Model Loaded")
    
    # Save explanations to your validation data
    valDF = save_local_explanations(explainer, valDF, val_X)
    print("Explanations Saved to Validation Data")
    
    # Save Global Explanations as a pandas dataframe
    globalExplanations = save_global_explanations(explainer, val_X)
    print("Global Explanations Saved as Pandas Dataframe")
    
    # Save your Validation Set Predictions
    valDF = make_classification_predictions(newModel, valDF, val_X, val_Y)
    print('Validation Predictions written to CSV file in logs')
    
    # Set your Time Zone 
    timeZone = pytz.timezone(args.pytz_time_zone)
    timeLocal = dt.datetime.now(timeZone).strftime('%Y-%m-%d')
    print('Time Zone Set')
    
    # Make Output Directory
    datastorePath = args.output_path + '/' + timeLocal
    os.makedirs(datastorePath, exist_ok=True) 
    print('Output Directory Created')
    
    # Upload Validation Data with Predictions to Datastore
    write_to_datastore(valDF, ws, datastore, datastorePath, "validationPredictions.csv", False)
    print('Predictions with Explanations for Validation Data Loaded to Datastore')
    
    # Save your Global Explanations
    write_to_datastore(globalExplanations, ws, datastore, datastorePath, "globalExplanations.csv", False)
    print('Global Expanations Loaded to Datastore')
    
    # Calculate main scoring metric for the validation dataset
    newModelScore = score_model(newModel, val_X, val_Y, scoringMethod)
    print('Predictions Made for Validation Data')
    print('Validation Set ' + args.scoring_metric + ' is ' + str(round(newModelScore, 2)))
    
    # Retrieve confidence interval for the cross validated scoring metric
    lowerBoundColumn = args.scoring_metric + ' Lower CI'
    upperBoundColumn = args.scoring_metric + ' Upper CI'
    lowerBound = metricsTransposed[lowerBoundColumn][0]
    upperBound = metricsTransposed[upperBoundColumn][0]
    print('Model ' + str(args.scoring_metric) + ' is ' + str(args.confidence_level*100) +\
          '% likely to actually fall between ' + str(lowerBound) + ' and ' + str(upperBound))
    
    # Compare confidence interval of cross validation training metric with validation metric
    if (((args.metric_goal == 'MAXIMIZE') & (newModelScore < lowerBound)) or\
       (((args.metric_goal == 'MINIMIZE') & (newModelScore > upperBound)))):
        print('Model performance on training data is significantly different from performance on validation data.')
        raise Exception("Models performs differently on training and validation data.\
                         Please check to see if your model is overfitting.  Validation Set "\
                        + args.scoring_metric + ' is ' + str(round(newModelScore, 2)) + ".  "\
                        + 'Model ' + str(args.scoring_metric) + ' is ' + str(args.confidence_level*100) +\
                          '% likely to actually fall between ' + str(lowerBound) + ' and ' + str(upperBound))
    else:
        print('Model is performing as expected on validation data.')
    
    # Check to see if previous model exists and compare it to new model
    modelDictionary = ws.models
    if args.model_name in modelDictionary.keys():
        print('Models Being Compared')
        oldModel = load_model(args.model_name)
        oldModelScore = score_model(oldModel, val_X, val_Y, scoringMethod)
        print('Previous Model Loaded and is being Compared to New Model')
        if newModelScore > oldModelScore:
            registerFlag = 1
        else:
            registerFlag = 0
    else:
        registerFlag = 1
        print('No Previous Models Found')
    
    # Set your model tags and description
    description = args.project_description
    explainTags = set_tags(['Algorithm', 'Project', 'Model Type', 'Explainer Type'],\
                            ['XGB', args.project_name, 'Explainer', 'Mimic'])
    trainTags = set_tags(['Algorithm', 'Project', 'Model Type'], ['XGB', args.project_name, 'Classification'])
    print("Model Tags and Description Assigned")
    
    # Register your new model and explainer model
    if registerFlag == 1:
        modelPath = modelFolder + 'saved_model'
        register_model(ws, args.model_name, modelPath, trainDataset, valDataset, description, trainTags)
        modelNameExplainer = args.model_name + '-Explainer'
        explainerPath = modelFolder + 'explainer_model'
        register_model(ws, modelNameExplainer, explainerPath, trainDataset, valDataset, description, explainTags)
    else:
        print('Old model outperforms new model and new model will not be registered.')
        
    # Remove files from compute cluster
    shutil.rmtree(datastorePath)

    # Remove model pickle file from compute cluster
    shutil.rmtree(modelFolder)

if __name__ == '__main__':
    init()
    print('Script Initialized')
    main()
    print('Script Finished')
