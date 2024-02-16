import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import time
from sklearn.model_selection import train_test_split
import os

def preprocess_data(features, target):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name, display_logs=True):
    y_pred = model.predict(X_test)
    
    # Check if -1 is one of the unique values in y_pred
    is_anonymous = -1 in np.unique(y_pred)
    
    # Convert predictions to binary labels (0 or 1)
    y_pred_binary = np.where(y_pred == 1, 0, 1)

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary)

    # Print metrics
    print(f'Accuracy of Isolation Forest for {model_name}: {accuracy:.4f}')
    print(f'Classification Report for {model_name}:\n{report}')

    # Print logs
    if display_logs:
        log_message = f'Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}, Model: {model_name}, Prediction: {"Anonymous" if is_anonymous else "Not Anonymous"}'
        print(log_message)

    # Print only anonymous logs
    if is_anonymous:
        print(log_message)

def new_data_available(directory='.'):
    # Specify the directory where new data is expected
    data_directory = os.path.join(os.getcwd(), directory)

    try:
        # Check if the path is a directory
        if os.path.isdir(data_directory):
            # Check if there are any files in the directory
            files = os.listdir(data_directory)
            return len(files) > 0
        else:
            return False
    except FileNotFoundError:
        return False

# Load datasets with explicit dtype specification
df_apache = pd.read_csv('newcsv.csv', dtype=str)
df_firewall = pd.read_csv('main_data.csv', dtype=str)

# Handle mixed-type columns using LabelEncoder and impute missing values
label_encoder = LabelEncoder()
imputer = SimpleImputer(strategy='most_frequent')

for col in df_apache.columns[df_apache.dtypes == 'object']:
    df_apache[col] = label_encoder.fit_transform(df_apache[col].astype(str))

for col in df_firewall.columns[df_firewall.dtypes == 'object']:
    df_firewall[col] = label_encoder.fit_transform(df_firewall[col].astype(str))

# Impute missing values
df_apache = pd.DataFrame(imputer.fit_transform(df_apache), columns=df_apache.columns)
df_firewall = pd.DataFrame(imputer.fit_transform(df_firewall), columns=df_firewall.columns)

# Preprocess Apache dataset
apache_features = df_apache.iloc[:, :-1]  # Exclude the target column
apache_target = df_apache.iloc[:, -1]
X_train_apache, X_test_apache, y_train_apache, y_test_apache = preprocess_data(apache_features, apache_target)

# Preprocess Firewall dataset
firewall_features = df_firewall.iloc[:, :-1]  # Exclude the target column
firewall_target = df_firewall.iloc[:, -1]
X_train_firewall, X_test_firewall, y_train_firewall, y_test_firewall = preprocess_data(firewall_features, firewall_target)

# Isolation Forest for Apache
iforest_apache = IsolationForest(contamination=0.1, random_state=42)
iforest_apache.fit(X_train_apache)

# Isolation Forest for Firewall
iforest_firewall = IsolationForest(contamination=0.1, random_state=42)
iforest_firewall.fit(X_train_firewall)

# Real-time processing loop
while True:
    print("Processing new data...")

    # Check if new data is available (replace this with your actual condition)
    if new_data_available():
        # Read new data
        new_data_apache = pd.read_csv('newcsv.csv', dtype=str)
        new_data_firewall = pd.read_csv('main_data.csv', dtype=str)

        # Handle mixed-type columns using LabelEncoder and impute missing values
        for col in new_data_apache.columns[new_data_apache.dtypes == 'object']:
            new_data_apache[col] = label_encoder.fit_transform(new_data_apache[col].astype(str))

        for col in new_data_firewall.columns[new_data_firewall.dtypes == 'object']:
            new_data_firewall[col] = label_encoder.fit_transform(new_data_firewall[col].astype(str))

        # Impute missing values
        new_data_apache = pd.DataFrame(imputer.fit_transform(new_data_apache), columns=new_data_apache.columns)
        new_data_firewall = pd.DataFrame(imputer.fit_transform(new_data_firewall), columns=new_data_firewall.columns)

        # Preprocess new data
        X_new_data_apache = new_data_apache.iloc[:, :-1]
        y_new_data_apache = new_data_apache.iloc[:, -1]

        X_new_data_firewall = new_data_firewall.iloc[:, :-1]
        y_new_data_firewall = new_data_firewall.iloc[:, -1]

        # Evaluate model on new data and display logs
        evaluate_model(iforest_apache, X_new_data_apache, y_new_data_apache, 'Apache')
        evaluate_model(iforest_firewall, X_new_data_firewall, y_new_data_firewall, 'Firewall')

    # Pause for a while before the next iteration
    time.sleep(60)  # Adjust this interval as needed
