import os
import argparse
from azureml.core import Run, Dataset, Datastore
from azureml.dataprep.api.engineapi.typedefinitions import Target
import pandas as pd
from sklearn.model_selection import train_test_split
from decimal import Decimal
from datetime import datetime

run = Run.get_context()
ws = run.experiment.workspace

parser = argparse.ArgumentParser()
parser.add_argument("--folder_name", type=str, help="directory of the storage name")
parser.add_argument("--file_name", type=str, help="filename that needs to be split into train and validate")
parser.add_argument("--datastore_name", type=str, help="datastore name")
parser.add_argument("--train_file_name", type=str, help="name of the file to save as training file")
parser.add_argument("--val_file_name", type=str, help="name of the file to save as the validation file")
parser.add_argument("--label_name", type=str, help="the column from the dataset that is the label")
parser.add_argument("--train_size", type=str, help="the percent that is to be used for training dataset")
args = parser.parse_args()

# parse out the parameters of the input values
folder_name = args.folder_name
file_name = args.file_name
datastore_name = args.datastore_name
label_name = args.label_name
train_size = args.train_size
train_file_name = args.train_file_name
val_file_name = args.val_file_name

# use regular expressions to determine if the string is a decimal
import re
regex = r'^[+-]{0,1}((\d*\.)|\d*)\d+$'

# get the train size and calculate out the test size
if re.match(regex, train_size) is None:
    raise Exception("Please provide a decimal value as a string")

if Decimal(train_size) > 0.85:
    raise Exception('Training size cannot be equal to or larger than 0.85')

train_size = Decimal(train_size)
test_size = Decimal("1.00") - Decimal(train_size)

# get the datastore and the tabular dataset
datastore = Datastore(ws, datastore_name)
path_on_datastore = os.path.join(folder_name, file_name)

dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, path_on_datastore)
)

# convert to pandas to split the data
data = dataset.to_pandas_dataframe()
X = data.drop(columns=label_name)
y = data[label_name]

# split the data using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=101)
# join train and train label; same for test
train = pd.concat([X_train, y_train], axis=1)
validation = pd.concat([X_test, y_test], axis=1)

# make sure folder_name was passed in as an argument
if not (folder_name is None):
    os.makedirs("files", exist_ok=True)
    print("%s created" % folder_name)

    # set the target path of the datastore to hold
    # test and validation datasets
    current_folder = str(datetime.now().date())

    target_path = os.path.join(folder_name, current_folder)

    train_file = os.path.join("files", train_file_name)
    val_file = os.path.join("files", val_file_name)
    # save the dataframes to the local drive to the upload the contents of thefolder
    train.to_csv(train_file,header=True, index=False)
    validation.to_csv(val_file,header=True, index=False)
    datastore.upload("files", target_path=target_path, overwrite=True, show_progress=False)

