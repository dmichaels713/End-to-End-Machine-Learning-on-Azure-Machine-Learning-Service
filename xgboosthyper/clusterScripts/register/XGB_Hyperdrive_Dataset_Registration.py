# Load in libaries
import argparse
import datetime as dt
import numpy as np
import os
import pandas as pd
import pytz
from itertools import zip_longest

# Load in Azure libraries
from azureml.core import Dataset, Datastore, Run, Workspace

# change the directory so the helper functions can be found
# cwd = os.getcwd()
# print('beginning root folder',cwd)

#def Test2(rootDir): 
#    for lists in os.listdir(rootDir): 
#        path = os.path.join(rootDir, lists) 
#        print(path)
#        if os.path.isdir(path): 
#            Test2(path)

#Test2(cwd)

# Load in functions from shared functions file
from util.XGB_Hyperdrive_Shared_Functions import create_dict, set_tags, write_to_datastore

# Define script-specific functions
def register_dataset(workspace, datastore, folder, file, datasetName, description, tags):
    filePath = os.path.join(folder, file)
    datastorePath = [(datastore, filePath)]
    dataset = Dataset.Tabular.from_delimited_files(datastorePath)
    dataset.register(workspace = workspace, 
                     create_new_version = True,
                     name = datasetName,
                     description = description,
                     tags = tags)
    
def init():
    # Set Arguments
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str,
                            help='Name of Training Dataset')
    parser.add_argument('--val_dataset_name', type=str,
                            help='Name of Validation Dataset')
    parser.add_argument('--datastore_path', type=str,
                            help='Location of file or files on Datastore')
    parser.add_argument('--datastore_name', type=str,
                            help='Name of Datastore')
    parser.add_argument('--train_file_name', type=str,
                            help='Name of training data file on Datastore')
    parser.add_argument('--val_file_name', type=str,
                            help='Name of validation data file on Datastore')
    parser.add_argument('--project_name', type=str,
                            help='Name of project')
    parser.add_argument('--project_description', type=str,
                            help='Description of project')
    parser.add_argument('--pytz_time_zone', type=str,
                            help='Time Zone associated with your data')
    args = parser.parse_args()

    # Set the Run context for logging
    global run
    run = Run.get_context()

def main():
    # Connect to your AMLS Workspace and set your Datastore
    ws = run.experiment.workspace
    datastoreName = args.datastore_name
    datastore = Datastore.get(ws, datastoreName)
    print('Datastore Set')
    
    # Set your Time Zone
    timeZone = pytz.timezone(args.pytz_time_zone)
    timeLocal = dt.datetime.now(timeZone).strftime('%Y-%m-%d')
    print('Time Zone Set')

    # Specify your File Names
    trainFile = timeLocal + '/' + args.train_file_name
    valFile = timeLocal + '/' + args.val_file_name
    print('File Names Set for Training and Validation Data.')
    
    # Set Tags and Description
    description = args.project_description
    trainTags = set_tags(['Project', 'Dataset Type', 'Date Created'],\
                         [args.project_name, 'Training', timeLocal])
    valTags = set_tags(['Project', 'Dataset Type', 'Date Created'],\
                       [args.project_name, 'Validation', timeLocal])
    print("Dataset Tags and Description Assigned")
    
    # Register your Training data as an Azure Tabular Dataset
    register_dataset(ws, datastore, args.datastore_path, trainFile, args.train_dataset_name, description, trainTags)
    print('Training Data Registered')
    
    # Register your Validation data as an Azure Tabular Dataset
    register_dataset(ws, datastore, args.datastore_path, valFile, args.val_dataset_name, description, valTags)
    print('Validation Data Registered')

if __name__ == '__main__':
    init()
    print('Script Initialized')
    main()
    print('Script Finished')
