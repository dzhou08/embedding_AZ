import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_validate
from pathlib import Path

import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger()

test_classification_result_folder = os.getenv("MODEL_TESTING_PATH_GENERAL")
test_classification_result_file = test_classification_result_folder + "/test_classification_results.csv"

def safe_eval(x):
    try:
        #logger.info(f"Converting embedding to array: ")
        # Remove the brackets and split the string into individual elements
        embedding_list = x.strip('[]').split(',')
        # Convert the list of strings to a NumPy array of floats
        return np.array(embedding_list, dtype=float)
    except Exception as e:
        logger.error(f"An error occurred while converting embedding to array: {e}")
        return np.array([])

# Turning the embeddings into a NumPy array, which will provide more flexibility in how to use it.
# It will also flatten the dimension to 1-D, which is the required format for many subsequent operations.
def embeddings_to_array(embeddings_file):
    df = pd.read_csv(embeddings_file)
    #print(f'df["embedding"]: {df["embedding"]}')
    df["embedding"] = df["embedding"].apply(safe_eval)
    df["linguistic_features"] = df["linguistic_features"].apply(safe_eval)
    #logger.info(df.head())
    return df


def cross_validation(model, _X, _y, _cv):
    # Define custom scoring metrics
    _scoring = {
        'accuracy': make_scorer(accuracy_score),  # How many predictions out of the whole were correct?
        'precision': make_scorer(precision_score, average='weighted'),  # How many out of the predicted
        # positives were actually positive?
        'recall': make_scorer(recall_score, average='weighted'),  # How many positive samples are captured
        # by the positive predictions?
        'f1_score': make_scorer(f1_score, average='macro')  # How balanced is the tradeoff between precision and recall?
    }

    scores = cross_validate(estimator=model,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    result = {}

    for metric in metrics:
        train_scores = scores[f'test_{metric}']
        train_scores_mean = round(train_scores.mean(), 3)
        train_scores_std = round(train_scores.std(), 3)

        test_scores = scores[f'test_{metric}']
        test_scores_mean = round(test_scores.mean(), 3)

        result[f'train_{metric}'] = train_scores
        result[f'train_{metric}_mean'] = train_scores_mean
        result[f'train_{metric}_std'] = train_scores_std

        result[f'test_{metric}'] = test_scores
        result[f'test_{metric}_mean'] = test_scores_mean

    return result

def clean_embedding(embedding_str):
    # Remove null bytes and any other unwanted characters
    if isinstance(embedding_str, str):
        return embedding_str.replace('\x00', '')
    return embedding_str

# Combine the specified columns into a single feature vector
def combine_features(row):
    # print column title of the row

    gaps_percentage = row["gaps_percentage"]
    disfluent_utterances = row["disfluent_utterances"]
    #combined_vector = np.concatenate(([gaps_percentage, disfluent_utterances], row["embedding"]))
    combined_vector = [gaps_percentage, disfluent_utterances]
    return combined_vector

def classify_embedding(train_data, test_data, _n_splits):
    """
    Perform classification using linguistic features (embeddings) from transcribed speech.

    Args:
        train_data (DataFrame): Training data containing 'diagnosis' labels and 'embedding' features.
                                Split into train and validation sets.
        test_data (DataFrame): Test data containing 'embedding' features.
        _n_splits (int): Number of folds for cross-validation.

    Returns:
        None

    This function trains and evaluates multiple machine learning models using the provided embeddings.
    It performs the following steps:
    - Loads training and test data.
    - Calculates baseline performance using a dummy stratified classifier.
    - Initializes machine learning models.
    - Creates cross-validation K-folds.
    - Iterates through each model:
        Model checking:
        - Performs hyperparameter optimization using cross-validation.
        - Records model performance on the training data using cross-validation.
        - Visualizes cross-validation results.
        Model building:
        - Trains the model on the entire training set with the best hyperparameters.
        - Predicts labels on the test data using the trained model.
        - Stores predictions in a CSV file.
        - Evaluates performance on the test data by comparing it to real diagnoses.
        - Records model sizes before and after training.
    - Saves performance results to a CSV file.
    - Saves model sizes to a CSV file.
    """
    logger.info("Initiating classification with GPT-3 text embeddings...")

    #print("before")
    # print train_data['linguistic_features']
    #print(f"train_data['linguistic_features']: {train_data['linguistic_features']}")
    # print train_data['embedding']
    #print(f"train_data['embedding']: {train_data['embedding']}")


    train_data["embedding"] = train_data["embedding"].apply(
        lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.zeros(1536)  # Adjust to the size of the embedding vector
    )
    # reduce the emdedding demension to 100
    # Set the desired number of components for PCA (e.g., 40 dimensions)
    pca = PCA(n_components=40)
    
    # Fit PCA on the embeddings and transform
    train_embeddings = np.array(train_data['embedding'].tolist())  # Shape: (num_samples, 1536)
    reduced_train_embeddings = pca.fit_transform(train_embeddings)
    train_data['reduced_embedding'] = reduced_train_embeddings.tolist()  # Each row is now a 100-dimensional list

    train_data["linguistic_features"] = train_data["linguistic_features"].apply(
        lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.zeros(4)  # Adjust size
    )

    #print("after")
    # print train_data['linguistic_features']
    #print(f"train_data['linguistic_features']: {train_data['linguistic_features']}")
    # print train_data['embedding']
    #print(f"train_data['embedding']: {train_data['embedding']}")
    #print(f"train_data['reduced_embedding']: {train_data['reduced_embedding']}")

    # Merge both embedding and linguistic_features columns
    train_data["merged_features"] = train_data.apply(
        lambda row: np.concatenate((row["reduced_embedding"], 
                                    row["linguistic_features"],
                                    #[float(row["gaps_percentage"]), float(row["disfluent_utterances"])]
                                    )), axis=1
    )

    ################### only for cha file approach ##################
    # Merge both embedding and linguistic_features columns
    # Combine the columns into a 6-element array
    train_data["combined_linguistic_features"] = train_data.apply(
        lambda row: np.concatenate([
            row["linguistic_features"],
            #[float(row["gaps_percentage"]), float(row["disfluent_utterances"])]
        ]), axis=1
    )

  
    # Define the dependent variable that needs to be predicted (labels)
    X_train = np.array(train_data["merged_features"].tolist())
    # Verify the shape of the result
    #print(f"Shape of X_train: {X_train.shape}")
    
    X_train = np.array(train_data["embedding"].tolist())

    # Define the independent variable 
    y_train = train_data['diagnosis'].values

    # Test data which is only used after training the model with the train data
    test_data["embedding"] = test_data["embedding"].apply(
        lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.zeros(1536)  # Adjust to the size of the embedding vector
    )
    test_data["linguistic_features"] = test_data["linguistic_features"].apply(
        lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.zeros(4)  # Adjust size
    )
    # Fit PCA on the embeddings and transform
    test_embeddings = np.array(test_data['embedding'].tolist())  # Shape: (num_samples, 1536)
    reduced_test_embeddings = pca.fit_transform(test_embeddings)
    #test_data['reduced_embedding'] = reduced_test_embeddings.tolist()  # Each row is now a 100-dimensional list
    test_data['reduced_embedding'] = test_embeddings.tolist()

    # Merge both embedding and linguistic_features columns
    test_data["merged_features"] = test_data.apply(
        lambda row: np.concatenate((row["reduced_embedding"], 
                                    #row["linguistic_features"],
                                    #[float(row["gaps_percentage"]), float(row["disfluent_utterances"])]
                                    )), axis=1
    )
    ################### only for cha file approach ##################
    # Merge both embedding and linguistic_features columns
    # Combine the columns into a 6-element array
    test_data["combined_linguistic_features"] = test_data.apply(
        lambda row: np.concatenate([
            row["linguistic_features"],
            #[float(row["gaps_percentage"]), float(row["disfluent_utterances"])]
        ]), axis=1
    )
    X_test = test_data['merged_features'].to_list()
    #print(f"Shape of X_train: {X_train.shape}")
    #print(f"Shape of X_test: {X_test.shape}")

    # Evaluate performance on test data
    # the ground truth real value for test diagnose is in the dx column
    test_ground_truth = pd.read_csv(os.getenv("GROUND_TRUTH_FILE"))
    # join the test_ground_truth and model_test_results on the ID and addressfname column
    # add test_ground_truth value into the model_test_results dataframe
    print(test_data)
    test_data = pd.merge(test_data, test_ground_truth, left_on='addressfname', right_on='ID')
    y_test = test_data['dx']#.to_list()



    baseline_score = dummy_stratified_clf(X_train, y_train)
    logger.debug(f"Baseline performance of the dummy classifier: {baseline_score}")

    # Create models
    models = [SVC(probability=True), 
              LogisticRegression(), 
              RandomForestClassifier()]
    names = ['SVC', 'LR', 'RF']

    # Split the dataset into k equal partitions (each partition is divided in train and validation data)
    cv = KFold(n_splits=_n_splits, random_state=42, shuffle=True)

    # Prepare dataframe for results
    results_df = pd.DataFrame(columns=['Set', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    models_size_df = pd.DataFrame(columns=['Model', 'Size'])

    logger.info("Beginning to train models using GPT embeddings...")

    # Collect total size of all models
    total_models_size = 0

    # report model classification results
    report_string = ""

    # Plot ROC curves for all models
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    # make a dictionary for color look up
    color_dict = {'SVC': 'blue', 'LR': 'red', 'RF': 'green'}

    for model, name in zip(models, names):
        logger.info(f"Initiating {name}...")

        ### Model checking
        best_params = hyperparameter_optimization(X_train, y_train, cv, model, name)
        model.set_params(**best_params)
        scores = cross_validation(model, X_train, y_train, cv)
        results_df = results_to_df(name, scores, results_df)

        # Visualize folds for different metrics in plots
        #visualize_results(_n_splits, name, scores, (test_classification_result_folder / "plots").resolve())

        ### Model building
        # Get size of the serialized model in bytes before training
        model_size = len(pickle.dumps(model, -1))
        logger.debug(f"Model size of {name} before training: {model_size} bytes.")

        # Train each model on the entire training set with best hyperparameters
        model.fit(X_train, y_train)
        # Evaluate on the test set
        # Predict on the test set
        y_pred = model.predict(X_test)

        print("y_pred")

        print(y_pred)
        print("y_test ")
        print(y_test)
        # Optionally, you can log the entire classification report
        report_string += f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}\n\n"

        # Get predicted probabilities or decision function scores
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]  # Probability for the positive class
        else:
            y_scores = model.decision_function(X_test)  # Decision function for SVC or similar

        #print(y_scores)
        
        # 5. Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        # 6. Compute the Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)

        # 7. Plot the ROC curve

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", color=color_dict[name])
        '''

        # Get size of the serialized model in bytes after training
        model_size = len(pickle.dumps(model, -1))
        total_models_size += model_size

        # Add trained model size to DataFrame
        models_size_df = pd.concat([models_size_df, pd.DataFrame([{'Model': name,
                                                                   'Size': f"{model_size} B",
                                                                   }])], ignore_index=True)

        # Load the empty task1 results CSV file
        model_test_results = pd.read_csv(test_classification_result_folder / 'classficiation_result.csv')

        # Predict label on test data with trained model
        model_predictions = model.predict(X_test)

        # Create a dictionary to store the filename-prediction value pairs
        filename_to_prediction = {}

        # Iterate through the filenames and model predictions arrays simultaneously
        for filename, prediction in zip(test_data['addressfname'], model_predictions):
            # Reverse binary classification
            filename_to_prediction[filename] = 'ProbableAD' if prediction == 1 else 'Control'

        # Fill the 'Prediction' column using the dictionary
        model_test_results['Prediction'] = model_test_results['ID'].map(filename_to_prediction)

        # Save the updated DataFrame in a new CSV file
        model_test_results_csv = (test_classification_result_folder / 'classficiation_result.csv').resolve()
        model_test_results.to_csv(model_test_results_csv, index=False)
        
        # print the length of the model_test_results
        #evaluate_similarity(name, model_test_results)
        '''

    plt.legend(loc="lower right")
    # Ensure test_classification_result_folder is a Path object
    test_classification_result_folder = Path(os.getenv("MODEL_TESTING_PATH_GENERAL"))
    plots_folder = test_classification_result_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    output_path = (plots_folder / 'plot_ROC_combined.png').resolve()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig(output_path)
    plt.show()
    logger.info(f"ROC plot saved to {output_path}")

    # save the classification report to a file
    report_file = (test_classification_result_folder / 'classification_report.txt').resolve()
    with open(report_file, 'w') as f:
        f.write(report_string)
    logger.info(f"Classification report saved to {report_file}")
    logger.info("Training using GPT embeddings done.")

    # Adjust resulting dataframe
    results_df = results_df.sort_values(by='Set', ascending=False)
    results_df = results_df.reset_index(drop=True)

    # Add baseline score to dataframe
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': 'Dummy',
                                                       'Accuracy': baseline_score,
                                                       }])], ignore_index=True)

    # Save results to csv
    embedding_results_file = (test_classification_result_folder / 'classficiation_result.csv').resolve()
    results_df.to_csv(embedding_results_file)
    logger.info(f"Writing {embedding_results_file}...")

    # Add total size to models_size
    logger.debug(f"Total size of all models: {total_models_size}.")
    models_size_df = pd.concat([models_size_df, pd.DataFrame([{'Model': 'Total',
                                                               'Size': f'{total_models_size} B',
                                                               }])], ignore_index=True)
    # Save results to csv
    models_size_file = test_classification_result_folder / 'models_size.csv'
    models_size_df.to_csv(models_size_file)
    logger.info(f"Writing {models_size_file}...")

    logger.info("Classification with text embeddings done.")


def evaluate_similarity(name, model_test_results):
    # Actual diagnosed data
    logger.info(f"Evaluating similarity between real and predicted diagnoses using model {name}...")
    logger.info(f"file ={test_classification_result_file}")
    test_results_task1 = pd.read_csv(test_classification_result_file)
    # print the length of the test_results_task1
    logger.info(f"Length of test_results_task1: {len(test_results_task1)}")

    real_diagnoses = test_results_task1['dx']
    predicted_diagnoses = model_test_results['Prediction']

    # Ensure both Series have the same index
    real_diagnoses = real_diagnoses.reset_index(drop=True)
    predicted_diagnoses = predicted_diagnoses.reset_index(drop=True)

    # Log the type and length of real_diagnoses and predicted_diagnoses
    #logger.info(f"real_diagnoses type: {type(real_diagnoses)}, length: {len(real_diagnoses)}")
    #logger.info(f"predicted_diagnoses type: {type(predicted_diagnoses)}, length: {len(predicted_diagnoses)}")
    # print the column header of the real_diagnoses
    #logger.info(f"real_diagnoses column header: {real_diagnoses}")
    # print the column header of the predicted_diagnoses
    #logger.info(f"predicted_diagnoses column header: {predicted_diagnoses}")

    # print all values in real_diagnoses
    #logger.info(f"real_diagnoses values: {real_diagnoses}")
    # print all values in predicted_diagnoses
    # map predicted_diagnoses to 0 if "Control" and 1 if "ProbableAD"
    predicted_diagnoses = predicted_diagnoses.map({'Control': 0, 'ProbableAD': 1})
    #logger.info(f"predicted_diagnoses values: {predicted_diagnoses}")

    # find out the matching record between the real and predicted diagnoses
    matching_values = (real_diagnoses == predicted_diagnoses).sum()
    # Calculate the total number of values
    total_values = len(real_diagnoses)
    # Calculate the percentage of matching values
    similarity_percentage = (matching_values / total_values) * 100
    logger.info(f"The similarity between the real and predicted diagnoses using model {name} "
                f"is {similarity_percentage:.2f}%.")

# Tune hyperparameters with GridSearchCV
def hyperparameter_optimization(X_train, y_train, cv, model, name):
    # Print shapes of X_train and y_train for debugging
    # print(f"X_train shape: {X_train}")
    # print(f"y_train shape: {y_train}")
    
    # Get the parameter grids
    lr_param_grid, rf_param_grid, svc_param_grid = param_grids()
    grid_search = None
    if name == 'SVC':
        grid_search = GridSearchCV(estimator=model, param_grid=svc_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
    elif name == 'LR':
        grid_search = GridSearchCV(estimator=model, param_grid=lr_param_grid, cv=cv, n_jobs=-1, error_score=0.0)
    elif name == 'RF':
        grid_search = GridSearchCV(estimator=model, param_grid=rf_param_grid, cv=cv, n_jobs=-1, error_score=0.0)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params


def param_grids():
    svc_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    lr_param_grid = [
        {'penalty': ['l1', 'l2'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['liblinear'],
         'max_iter': [100, 200, 500, 1000]},
        {'penalty': ['l2'],
         'C': np.logspace(-4, 4, 20),
         'solver': ['lbfgs'],
         'max_iter': [200, 500, 1000]},
    ]
    rf_param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }
    return lr_param_grid, rf_param_grid, svc_param_grid


def visualize_results(_n_splits, name, results, save_dir):
    plot_accuracy_path = (save_dir / f'plot_accuracy_{name}.png').resolve()
    plot_precision_path = (save_dir / f'plot_precision_{name}.png').resolve()
    plot_recall_path = (save_dir / f'plot_recall_{name}.png').resolve()
    plot_f1_path = (save_dir / f'plot_f1_{name}.png').resolve()
    # Plot Accuracy Result
    plot_result(name,
                "Accuracy",
                f"Accuracy scores in {_n_splits} Folds",
                results["train_accuracy"],
                results["test_accuracy"],
                plot_accuracy_path)
    # Plot Precision Result
    plot_result(name,
                "Precision",
                f"Precision scores in {_n_splits} Folds",
                results["train_precision"],
                results["test_precision"],
                plot_precision_path)
    # Plot Recall Result
    plot_result(name,
                "Recall",
                f"Recall scores in {_n_splits} Folds",
                results["train_recall"],
                results["test_recall"],
                plot_recall_path)
    # Plot F1-Score Result
    plot_result(name,
                "F1",
                f"F1 Scores in {_n_splits} Folds",
                results["train_f1_score"],
                results["test_f1_score"],
                plot_f1_path)


def results_to_df(name, scores, results_df):
    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Train',
                                                       'Model': name,
                                                       'Accuracy': f"{scores['train_accuracy_mean']} "
                                                                   f"({scores['train_accuracy_std']})",
                                                       'Precision': f"{scores['train_precision_mean']} "
                                                                    f"({scores['train_precision_std']})",
                                                       'Recall': f"{scores['train_recall_mean']} "
                                                                 f"({scores['train_recall_std']})",
                                                       'F1': f"{scores['train_f1_score_mean']} "
                                                             f"({scores['train_f1_score_std']})",
                                                       }])], ignore_index=True)

    results_df = pd.concat([results_df, pd.DataFrame([{'Set': 'Test',
                                                       'Model': name,
                                                       'Accuracy': scores['test_accuracy_mean'],
                                                       'Precision': scores['test_precision_mean'],
                                                       'Recall': scores['test_recall_mean'],
                                                       'F1': scores['test_f1_score_mean']
                                                       }])], ignore_index=True)
    return results_df


# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data, savefig_path=None):
    """Function to plot a grouped bar chart showing the training and validation
      results of the ML model in each fold after applying K-fold cross-validation.
     Parameters
     ----------
     x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'

     y_label: str,
        Name of metric being visualized e.g 'Accuracy'
     plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'

     train_data: list, array
        This is the list containing either training precision, accuracy, or f1 score.

     val_data: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
     savefig_path: str
        Save figures to this path if not empty (by default)
     Returns
     -------
     The function returns a Grouped Barchart showing the training and validation result
     in each fold.
    """

    # Set size of plot
    fig = plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold",
              "9th Fold", "10th Fold"]
    X_axis = np.arange(len(labels))
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    if savefig_path is not None:
        fig.savefig(savefig_path, dpi=fig.dpi)


def dummy_stratified_clf(X, y):
    """
    DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more complex classifiers.
    It gives us a measure of “baseline” performance — i.e. the success rate one should expect to achieve even
    if simply guessing.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    stratified_clf = DummyClassifier(strategy='stratified').fit(X_train, y_train)

    score = round(stratified_clf.score(X_test, y_test), 3)

    return score
