# Load in Libraries
import argparse
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel
from itertools import zip_longest
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import ShuffleSplit, cross_validate
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier

# Load in Azure libraries
from azureml.core import Run, Dataset, Workspace, Experiment
from azureml.interpret import ExplanationClient

# change the directory so the helper functions can be found

# Load in functions from shared functions file
from util.XGB_Hyperdrive_Shared_Functions import create_dict, save_local_explanations, make_classification_predictions
from util.XGB_Hyperdrive_Shared_Functions import split_x_y, save_global_explanations

# Define script-specific functions
def score_log_classification_training_data(model, features, target_column, cv_splits, bootstrap_sample_number):
    metrics_cv = cross_validate(model, features, target_column,\
                                scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1",], cv=cv_splits)
    
    # Get average of each metric across cross validation splits
    accuracy = np.mean(metrics_cv['test_accuracy'])
    balanced_accuracy = np.mean(metrics_cv['test_balanced_accuracy'])
    precision = np.mean(metrics_cv['test_precision'])
    recall = np.mean(metrics_cv['test_recall'])
    F1 = np.mean(metrics_cv['test_f1'])
    
    # Calculate Confidence Intervals for each of the metrics via bootstrapping cross-validated means
    resampled_mean_accuracy = []
    resampled_mean_balanced_accuracy = []
    resampled_mean_precision = []
    resampled_mean_recall = []
    resampled_mean_F1 = []
    metricsDF = pd.DataFrame(metrics_cv)
    
    for i in range(0, bootstrap_sample_number):
        resample = metricsDF.sample(frac=1, replace=True)
        mean_accuracy = np.mean(resample['test_accuracy'])
        mean_balanced_accuracy = np.mean(resample['test_balanced_accuracy'])
        mean_precision = np.mean(resample['test_precision'])
        mean_recall = np.mean(resample['test_recall'])
        mean_F1 = np.mean(resample['test_f1'])
        resampled_mean_accuracy.append(mean_accuracy)
        resampled_mean_balanced_accuracy.append(mean_balanced_accuracy)
        resampled_mean_precision.append(mean_precision)
        resampled_mean_recall.append(mean_recall)
        resampled_mean_F1.append(mean_F1)
    resampled_mean_accuracy.sort()
    resampled_mean_balanced_accuracy.sort()
    resampled_mean_precision.sort()
    resampled_mean_recall.sort()
    resampled_mean_F1.sort()
    lower_bound_index = int(np.floor(bootstrap_sample_number*(1-args.confidence_level)/2))
    upper_bound_index = int(np.floor(bootstrap_sample_number*(1+args.confidence_level)/2))
    
    print("Scoring Done for Training Data")
    
    # Log training metrics
    run.log('Accuracy Training', np.float(accuracy))
    run.log('Balanced Accuracy Training', np.float(balanced_accuracy))
    run.log('Recall Training', np.float(recall))
    run.log('Precision Training', np.float(precision))
    run.log('F1 Training', np.float(F1))
    run.log('Accuracy Training Lower CI', np.float(resampled_mean_accuracy[lower_bound_index]))
    run.log('Balanced Accuracy Training Lower CI', np.float(resampled_mean_balanced_accuracy[lower_bound_index]))
    run.log('Recall Training Lower CI', np.float(resampled_mean_recall[lower_bound_index]))
    run.log('Precision Training Lower CI', np.float(resampled_mean_precision[lower_bound_index]))
    run.log('F1 Training Lower CI', np.float(resampled_mean_F1[lower_bound_index]))
    run.log('Accuracy Training Upper CI', np.float(resampled_mean_accuracy[upper_bound_index]))
    run.log('Balanced Accuracy Training Upper CI', np.float(resampled_mean_balanced_accuracy[upper_bound_index]))
    run.log('Recall Training Upper CI', np.float(resampled_mean_recall[upper_bound_index]))
    run.log('Precision Training Upper CI', np.float(resampled_mean_recall[upper_bound_index]))
    run.log('F1 Training Upper CI', np.float(resampled_mean_F1[upper_bound_index]))
    run.log_list('Accuracy for all CV Splits', metrics_cv['test_accuracy'])
    run.log_list('Balanced Accuracy for all CV Splits', metrics_cv['test_balanced_accuracy'])
    run.log_list('Precision for all CV Splits', metrics_cv['test_precision'])
    run.log_list('Recall for all CV Splits', metrics_cv['test_recall'])
    run.log_list('F1 for all CV Splits', metrics_cv['test_f1'])
    return print("Metrics Logged for Training Data")
    
def score_log_classification_validation_data(classificationModel, features, target_column):
    val_binary_predictions = classificationModel.predict(features)
    val_accuracy = accuracy_score(target_column, val_binary_predictions)
    val_balanced_accuracy = balanced_accuracy_score(target_column, val_binary_predictions)
    val_precision = precision_score(target_column, val_binary_predictions, average='micro')
    val_recall = recall_score(target_column, val_binary_predictions, average='micro')
    val_F1 = f1_score(target_column, val_binary_predictions, average='micro')
    print("Scoring Done for Validation Data")
    
    run.log('Accuracy Validation', np.float(val_accuracy))
    run.log('Balanced Accuracy Validation', np.float(val_balanced_accuracy))
    run.log('Recall Validation', np.float(val_recall))
    run.log('Precision Validation', np.float(val_precision))
    run.log('F1 Validation', np.float(val_F1))
    return print("Metrics Logged for Validation Data")

def log_classification_charts(dataType, classificationModel, features, targetColumn):
    # Get predictions and actuals to make Confusion Matrix and Precision Recall Curve
    binary_predictions = classificationModel.predict(features)
    probability_predictions = classificationModel.predict_proba(features)[:,1]
    print("Predictions Made for " + dataType + " Data Charts")
    
    # Log a Confusion Matrix
    data = pd.DataFrame(dict(s1 = targetColumn, s2 = binary_predictions)).reset_index()
    confusion_matrix = pd.crosstab(data['s1'], data['s2'], rownames=['Actual'], colnames=['Predicted'])
    fig = plt.figure()
    sns.heatmap(confusion_matrix, annot=True,cmap='Blues', fmt='g')
    plt.title("Confusion Matrix " + dataType)
    plt.close(fig)
    run.log_image(name='confusion-matrix-' + dataType.lower(), plot=fig)
    print("Confusion Matrix Logged for " + dataType + " Data")

    # Log a Precision / Recall Curve
    lr_precision, lr_recall, _ = precision_recall_curve(targetColumn, probability_predictions)
    lr_f1, lr_auc = f1_score(targetColumn, binary_predictions, average='micro'), auc(lr_recall, lr_precision)
    no_skill = len(targetColumn[targetColumn==1]) / len(targetColumn)
    fig2 = plt.figure()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    plt.title('Precision Recall Curve ' + dataType)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.close(fig2)
    run.log_image(name='precision-recall-curve-' + dataType.lower(), plot=fig2)
    print("Precision Recall Curve Logged for " + dataType + " Data")
    
    # Log a Receiving Operating Characteristic (ROC) Curve  
    fpr, tpr, thresholds = roc_curve(targetColumn, probability_predictions) 
    fig3 = plt.figure()
    plt.plot(fpr, tpr, color='lightblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + dataType)
    plt.legend()
    plt.close(fig3)
    run.log_image(name='roc-curve-' + dataType.lower(), plot=fig3)
    print("ROC Curve Logged for " + dataType + " Data")
    
def init():
    # Set Arguments.  These should be all of the hyperparameters you will tune.
    global args
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Learning Rate')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning Rate')
    parser.add_argument('--scale_pos_weight', type=float, default=0.6,
                        help='Helps with Unbalanced Classes.  Should be Sum(Negative)/Sum(Positive)')
    parser.add_argument('--booster', type=str, default='gbtree',
                        help='The type of Boosting Algorithim')
    parser.add_argument('--min_child_weight', type=float, default=1,
                        help='Controls Overfitting')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Controls Overfitting')
    parser.add_argument('--gamma', type=float, default=0,
                        help='Make Algorithm Conservative')
    parser.add_argument('--subsample', type=float, default=1,
                        help='Controls Overfitting')
    parser.add_argument('--colsample_bytree', type=float, default=1,
                        help='Defines Sampling')
    parser.add_argument('--reg_lambda', type=float, default=1,
                        help='Controls Overfitting')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Reduces Dimensionality')
#    parser.add_argument('--objective', type=str, default='binary:logistic',reg:logistic,multi:softmax
    parser.add_argument('--objective', type=str, default='multi:softmax',
                        help='Defines Training Objective Metric')
    # Other Parameters
    parser.add_argument('--train_dataset_name', type=str,
                        help='Name of Training Dataset')
    parser.add_argument('--val_dataset_name', type=str,
                        help='Name of Validation Dataset')
    parser.add_argument('--target_column_name', type=str,
                        help='Name of variable to score')
    parser.add_argument('--k_folds', type=int, default = 10,
                        help='Number of folds to split your data into for cross validation')
    parser.add_argument('--shuffle_split_size', type=float,
                        help='Percentage of data to hold out for testing during cross validation')
    parser.add_argument('--confidence_level', type=float, default = 0.95,
                        help='Level of confidence to set for your confidence interval ()')
    args = parser.parse_args()
    print(args)

    # Set the Run context for logging
    global run
    run = Run.get_context()
    
    # log your hyperparameters,
    run.log('eta',np.float(args.eta))
    run.log('learning_rate',np.float(args.learning_rate))
    run.log('scale_pos_weight',np.float(args.scale_pos_weight))
    run.log('booster',np.str(args.booster))
    run.log('min_child_weight',np.float(args.min_child_weight))
    run.log('max_depth',np.float(args.max_depth))
    run.log('gamma',np.float(args.gamma))
    run.log('subsample',np.float(args.subsample))
    run.log('colsample_bytree',np.float(args.colsample_bytree))
    run.log('reg_lambda',np.float(args.reg_lambda))
    run.log('alpha',np.float(args.alpha))
    run.log('objective',np.str(args.objective))

# Write your main function.  This will train and log your model.
def main():
    # Connect to your AMLS Workspace and retrieve your data
    ws = run.experiment.workspace
    training_dataset_name  = args.train_dataset_name
    train_dataset  = Dataset.get_by_name(ws, training_dataset_name, version='latest')
    val_dataset_name  = args.val_dataset_name
    val_dataset  = Dataset.get_by_name(ws, val_dataset_name, version='latest')
    print('Datasets Retrieved')
    
    # Transform your data to Pandas
    trainTab =  train_dataset
    trainDF = trainTab.to_pandas_dataframe()
    valTab =  val_dataset
    valDF = valTab.to_pandas_dataframe()
    print('Datasets Converted to Pandas')
    
    # Split out X and Y variables for both training and validation data
    X, Y = split_x_y(trainDF, args.target_column_name)
    val_X, val_Y = split_x_y(valDF, args.target_column_name)
    print("Data Ready for Scoring")
 
    # Set your model and hyperparameters
    hyperparameters = dict(eta=args.eta,\
                           learning_rate=args.learning_rate,\
                           scale_pos_weight=args.scale_pos_weight,\
                           booster = args.booster,\
                           min_child_weight = args.min_child_weight,\
                           max_depth = args.max_depth,\
                           gamma = args.gamma,\
                           subsample = args.subsample,\
                           colsample_bytree = args.colsample_bytree,\
                           reg_lambda = args.reg_lambda,\
                           alpha = args.alpha,\
                           num_class=5,
                           objective = args.objective)
    
    model = XGBClassifier(**hyperparameters)
    print('Hyperparameters Set')
    
    # Fit your model
    xgbModel = model.fit(X,Y)
    print("Model Fit")
    
    # Score your training data with cross validation and log metrics
    ss = ShuffleSplit(n_splits=args.k_folds, test_size=args.shuffle_split_size, random_state = 33)
    bootstrap_sample_number = args.k_folds*100
    score_log_classification_training_data(model, X, Y, ss, bootstrap_sample_number)
    
    # Log a Confusion Matrix and Precision Recall Curve for your training data
    log_classification_charts("Training", xgbModel, X, Y)
    
    # Score your validation data and log metrics
    score_log_classification_validation_data(xgbModel, X, Y)
    print("Scoring Done for Validation Data")

    # Log a Confusion Matrix and Precision Recall Curve for your training data
    log_classification_charts("Validation", xgbModel, val_X, val_Y)
    
    # Model Explanations
    client = ExplanationClient.from_run(run)
    explainer = MimicExplainer(xgbModel, 
                               X, 
                               LGBMExplainableModel,
                               classes = list(val_Y.unique()),
                               features = val_X.columns,
                               shap_values_output = 'probability',
                               model_task = 'classification')
    global_explanation = explainer.explain_global(X)
    print(global_explanation)
    client.upload_model_explanation(global_explanation, top_k=30)
    print("Global Explanations Created")
    
    # Save local Explanations in json format to a column in the Validation Set
    valDF = save_local_explanations(explainer, valDF, val_X)
    print("Explanations Saved to Validation Data")
    
    # Save Global Explanations as a pandas dataframe
    globalExplanations = save_global_explanations(explainer, val_X)
    print("Global Explanations Saved as Pandas Dataframe")
    
    # Make a folder in which to save your output
    os.makedirs('outputs', exist_ok=True)
    
    # Save your Model
    joblib.dump(xgbModel, 'outputs/XGBmodel.pkl')
    print("Model Saved")
    
    # Save your Explainer Model
    joblib.dump(explainer, 'outputs/LGBMexplainer.pkl')
    print("Explainer Model Saved")
    
    # Save your Validation Set Predictions
    valDF = make_classification_predictions(xgbModel, valDF, val_X, val_Y)
    valCSV = valDF.to_csv('outputs/validationPredictions.csv', index=False)
    print('Validation Predictions written to CSV file in logs')
    
    # Save your Global Explanations
    globalExplanationsCSV = globalExplanations.to_csv('outputs/globalExplanations.csv', index=False)
    print('Global Explanations written to CSV file in logs')

if __name__ == '__main__':
    init()
    print('Script Initialized')
    main()
    print('Script Finished')
