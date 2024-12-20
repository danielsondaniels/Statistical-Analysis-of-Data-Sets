import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_describe_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    return df

def compute_descriptive_stats(df):
    print("\nNumerical Attribute Statistics:")
    print(df.describe())
    print("\nCategorical Attribute Mode and Frequency:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\nColumn: {col}")
        print("Mode:", df[col].mode()[0])
        print("Frequency:\n", df[col].value_counts())

def data_visualization(df):
    df.hist(figsize=(12, 8), bins=20)
    plt.suptitle("Histograms for Numerical Attributes")
    plt.show()
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
    plt.title("Boxplots for Numerical Attributes")
    plt.show()
    for col in df.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"Bar Chart for {col}")
        plt.show()
    sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
    plt.suptitle("Pairwise Relationship Plots", y=1.02)
    plt.show()

def handle_missing_data(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    print("\nMissing values after handling:\n", df.isnull().sum())
    return df

def detect_outliers(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\nColumn: {col} | Outliers Detected: {len(outliers)}")
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def correlation_analysis(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()

def data_transformation(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df = pd.get_dummies(df, drop_first=True)
    print("\nData after transformation:\n", df.head())
    return df

def main():
    file_path = "dataset.csv"
    df = load_and_describe_dataset(file_path)
    compute_descriptive_stats(df)
    data_visualization(df)
    df = handle_missing_data(df)
    df = detect_outliers(df)
    correlation_analysis(df)
    df = data_transformation(df)
    df.to_csv("cleaned_dataset.csv", index=False)
    print("\nCleaned and transformed dataset saved as 'cleaned_dataset.csv'.")

if __name__ == "__main__":
    main()
