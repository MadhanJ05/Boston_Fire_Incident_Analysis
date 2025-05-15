import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data(Users/baibai/Desktop/Dataset.csv):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Perform preprocessing: calculate total loss and classify high-loss events.
    """
    df["total_loss"] = df["estimated_property_loss"] + df["estimated_content_loss"]
    df["high_loss"] = (df["total_loss"] > 1000).astype(int)  
    return df

def analyze_high_loss(df):
    """
    Print high-loss event proportion.
    """
    print(df["high_loss"].value_counts(normalize=True) * 100)

def preprocess_data(df):
    """
    Preprocess the dataset by encoding categorical variables, 
    converting date/time columns, dropping unnecessary columns, 
    and splitting into training/testing sets.
    """
    
    categorical_cols = ["incident_type", "district", "incident_description", "property_use"]
    label_encoders = {}

    for col in categorical_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    
    df["alarm_date"] = pd.to_datetime(df["alarm_date"], errors="coerce")
    df["alarm_time"] = pd.to_datetime(df["alarm_time"], format="%H:%M:%S", errors="coerce")

    df["year"] = df["alarm_date"].dt.year
    df["month"] = df["alarm_date"].dt.month
    df["day_of_week"] = df["alarm_date"].dt.dayofweek
    df["hour"] = df["alarm_time"].dt.hour

    
    drop_cols = ["incident_number", "alarm_date", "alarm_time", "total_loss"]
    df.drop(columns=drop_cols, inplace=True)

    X = df.drop(columns=["high_loss"])
    y = df["high_loss"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df


def save_cleaned_data(df, filename="Users/baibai/Desktop/cleaned_dataset.csv"):
    """
    Saves the cleaned dataset as a CSV file.
    """
    df.to_csv(filename, index=False)
    print("✅ Cleaned dataset saved successfully!")

def encode_categorical_columns(df):
    """
    Identifies and encodes categorical columns using Label Encoding.
    Returns the modified dataframe and a dictionary of label encoders.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    print("Categorical columns:", categorical_cols)


def check_non_numeric_columns(df):
    """
    Identifies non-numeric columns in the given dataframe.
    Prints the column names and their unique values.
    """
    non_numeric_cols = df.select_dtypes(include=["object"]).columns
    print("🔍 Non-numeric columns in dataset:", non_numeric_cols)

    for col in non_

def encode_categorical_columns(df):
    """
    Converts categorical columns to numeric values using Label Encoding.
    Returns the modified dataframe and the encoders for future use.
    """
    non_numeric_cols = [
        "city_section", "neighborhood", "zip", "property_description",
        "street_number", "street_prefix", "street_name", "street_suffix",
        "street_type", "address_2", "xstreet_prefix", "xstreet_name",
        "xstreet_suffix", "xstreet_type"
    ]

    label_encoders = {}
    for col in non_numeric_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # Store encoders if we need to decode later

    print("✅ All categorical columns have been converted to numeric values!")
    return df, labe


def encode_categorical_columns(X_train, X_test):
    """
    Encodes categorical columns and handles unseen labels in test data.
    
    Parameters:
        X_train (DataFrame): Training dataset
        X_test (DataFrame): Testing dataset
    
    Returns:
        X_train, X_test (DataFrames): Transformed datasets with encoded categorical values
    """
    non_numeric_cols_train = X_train.select_dtypes(include=["object"]).columns
    non_numeric_cols_test = X_test.select_dtypes(include=["object"]).columns

    print("🔍 Non-numeric columns in X_train:", non_numeric_cols_train)
    print("🔍 Non-numeric columns in X_test:", non_numeric_cols_test)

    for col in non_numeric_cols_train:
        le = LabelEncoder()
        
        # Fit on train and transform both train & test
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        # Convert unseen categories in X_test to 'Unknown'
        X_test[col] = X_test[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')

        # Add 'Unknown' category and transform test set
        le.classes_ = np.append(le.classes_, 'Unknown')
        X_test[col] = le.transform(X_test[col])

    print("✅ All categorical columns encoded and unseen labels handled!")
    
    return X_train, X_test


def encode_single_column(X_train, X_test, column_name):
    """
    Encodes a single categorical column and handles unseen labels in the test set.

    Parameters:
        X_train (DataFrame): Training dataset
        X_test (DataFrame): Testing dataset
        column_name (str): The name of the categorical column to encode

    Returns:
        X_train, X_test (DataFrames): Transformed datasets with encoded values for the specified column
    """
    le = LabelEncoder()

    # Fit on train and transform both train & test
    X_train[column_name] = le.fit_transform(X_train[column_name].astype(str))
    X_test[column_name] = X_test[column_name].astype(str).apply(lambda x: x if x in le.classes_ else "Unknown")

    # Add 'Unknown' category and transform test set
    le.classes_ = np.append(le.classes_, "Unknown")
    X_test[column_name] = le.transform(X_test[column_name])

    print(f"✅ '{column_name}' column encoded successfully!")
    
    return X_train, X_test


def preprocess_data(df):
    df["total_loss"] = df["estimated_property_loss"] + df["estimated_content_loss"]
    df["high_loss"] = (df["total_loss"] > 1000).astype(int)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  

    return df, label_encoders

def split_data(df):
    X = df.drop(columns=["high_loss"])
    y = df["high_loss"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low Loss (0)", "High Loss (1)"])
    return accuracy, report

if __name__ == "__main__":
    df = load_data("Dataset.csv")
    df, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("\n📌 Classification Report:\n", report)


def train_tuned_model(X_train, y_train):
    """
    Train a tuned Random Forest model with hyperparameter regularization.
    """
    rf_tuned = RandomForestClassifier(
        n_estimators=50,       
        max_depth=10,          
        min_samples_split=10, 
        min_samples_leaf=5,    
        max_features="sqrt",   
        random_state=42,
        n_jobs=-1,            
        class_weight="balanced"
    )

    rf_tuned.fit(X_train, y_train)
    return rf_tuned


def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform cross-validation and return mean accuracy.
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"📌 Cross-validation scores: {cv_scores}")
    print(f"📌 Mean CV Accuracy: {cv_scores.mean():.4f}")
    return cv_scores.mean()


def encode_incident_type(df):
    """
    Encode the 'incident_type' column and return the mapping dictionary.
    """
    le = LabelEncoder()
    df["incident_type"] = le.fit_transform(df["incident_type"])
    incident_type_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    
    print("📌 Incident Type Mapping:")
    print(incident_type_mapping)
    
    return df, incident_type_mapping

def preprocess_data(df):
    df["total_loss"] = df["estimated_property_loss"] + df["estimated_content_loss"]
    df["high_loss"] = (df["total_loss"] > 1000).astype(int)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df.drop(columns=["incident_number", "alarm_date", "alarm_time", "total_loss"], inplace=True, errors="ignore")
    return df

def split_data(df):
    X = df.drop(columns=["high_loss"])
    y = df["high_loss"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    return model

def compute_feature_importance(model, X_train):
    """
    Compute feature importance from a trained model.
    
    Args:
    - model: Trained RandomForest model
    - X_train: Training dataset (features only)
    
    Returns:
    - feature_importances: DataFrame with feature names and importance scores
    """
    feature_importances = pd.DataFrame({
        "Feature": X_train.columns[:len(model.feature_importances_)],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return feature_importances


def save_data(df, filepath):
    """ Save cleaned dataset to CSV. """
    df.to_csv(filepath, index=False)
    print(f"✅ Cleaned dataset saved successfully at {filepath}!")


def preprocess_data(df):
    """ Preprocess dataset: clean, encode categorical variables, and create new features. """
    
    
    df["total_loss"] = df["estimated_property_loss"] + df["estimated_content_loss"]

    df["high_loss"] = (df["total_loss"] > 1000).astype(int)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    print("✅ All categorical columns converted to numeric!")
    return df, label_encoders


def train_model(df):
    """ Train Random Forest model for high-loss fire event classification. """
    X = df.drop(columns=["high_loss"])
    y = df["high_loss"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Loss (0)", "High Loss (1)"]))
    
    return model, X_train, X_test, y_train, y_test


def plot_feature_importance(model, X_train):
    """ Plot top 10 important features in predicting high-loss fire events. """
    feature_importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"][:10], feature_importances["Importance"][:10])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Important Features for Predicting High Loss")
    plt.gca().invert_yaxis()
    plt.show()

def preprocess_data(df):
    df["total_loss"] = df["estimated_property_loss"] + df["estimated_content_loss"]
    df["high_loss"] = (df["total_loss"] > 1000).astype(int)

    categorical_cols = ["incident_type", "district", "incident_description", "property_use"]
    label_encoders = {}
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def split_data(df):
    X = df.drop(columns=["high_loss"])
    y = df["high_loss"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low Loss (0)", "High Loss (1)"])
    return accuracy, report

def get_feature_importance(model, X_train):
    feature_importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    return feature_importances

def plot_feature_importance(feature_importances):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"][:10], feature_importances["Importance"][:10])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Important Features for Predicting High Loss")
    plt.gca().invert_yaxis()
    plt.show()


def preprocess_data(df):
    df["total_loss"] = df["estimated_property_loss"] + df["estimated_content_loss"]
    df["high_loss"] = (df["total_loss"] > 1000).astype(int)
    
    categorical_cols = ["incident_type", "district", "incident_description", "property_use"]
    label_encoders = {}
    
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def plot_feature_importance(model, feature_names):
    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"][:10], feature_importances["Importance"][:10])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Important Features for Predicting High Loss")
    plt.gca().invert_yaxis()
    plt.show()



