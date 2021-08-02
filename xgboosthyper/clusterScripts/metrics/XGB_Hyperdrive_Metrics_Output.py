# Load in libraries
import argparse
import datetime as dt
import json
import os
import pandas as pd
import pytz
import shutil
from itertools import zip_longest
from shutil import copy2

# Load in Azure libraries
from azureml.core import Dataset, Datastore, Experiment, Model, Run, Workspace


# Load in functions from shared functions file
from util.XGB_Hyperdrive_Shared_Functions import write_to_datastore

def init():
    # Set Arguments.
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastore_name', type=str,
                            help='Name of Datastore')
    parser.add_argument('--pytz_time_zone', type=str,
                            help='Time Zone associated with your data')
    parser.add_argument('--output_path', type=str,
                            help='Location to store output on Datastore')
    parser.add_argument('--scoring_metric', type=str,
                            help='Metric with which you scored your Hyperdrive Run')
    parser.add_argument('--metrics_data', type=str,
                            help='Location of Hyperdrive Run Metrics File')
    args = parser.parse_args()

    # Set the Run context for logging
    global run
    run = Run.get_context()

def main():   
    # Retrieve your Metrics Data file
    with open(args.metrics_data) as metrics:
        metricsData = json.load(metrics)
    print('Metrics File Downloaded')
    
    # Turn the Metrics JSON file into two pandas dataframes
    metrics = pd.DataFrame(metricsData)
    metricsTransposed = metrics.transpose().sort_values(by=args.scoring_metric, ascending=False)
    print('Metrics Dataframes Created')
    
    # Connect to your AMLS Workspace and set your Datastore
    ws = run.experiment.workspace
    datastore = Datastore.get(ws, args.datastore_name)
    print('Datastore Set')
    
    # Set your Time Zone 
    timeZone = pytz.timezone(args.pytz_time_zone)
    timeLocal = dt.datetime.now(timeZone).strftime('%Y-%m-%d')
    print('Time Zone Set')
    
    # Make Output Directory
    outputFolder = args.output_path + '/' + timeLocal
    os.makedirs(outputFolder, exist_ok=True) 
    print('Output Directory Created')
    
    # Upload csv files to Datastore
    write_to_datastore(metrics, ws, datastore, outputFolder, 'Metrics.csv', True)
    write_to_datastore(metricsTransposed, ws, datastore, outputFolder, 'MetricsTransposed.csv', True)
    print('Hyperdrive Metrics Data Loaded to Datastore')
    
    # Remove files from compute cluster
    shutil.rmtree(outputFolder)


if __name__ == '__main__':
    init()
    print('Script Initialized')
    main()
    print('Script Finished')
