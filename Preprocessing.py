import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
import os
from datetime import datetime

class Preprocessing:
    """
    A class for data preprocessing, including loading, handling duplicates,
    handling missing values, encoding, and outlier processing.
    """

    def __init__(self, data_path, output_dir):
        """
        Initializes the Preprocessing object.
        Args:
            data_path (str): Path to the CSV file containing the dataset.
            output_dir (str): Directory to save the processed CSV file and visualizations.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None

    def run_pipeline(self):
        """
        Executes the entire preprocessing pipeline and saves the output to a CSV file.
        """
        print("\n--- Starting Data Preprocessing Pipeline ---")
        start_time = time.time()

        print("Step 1: Loading data...")
        self.load_data()

        print("Step 2: Handling duplicates...")
        self.df = self.handle_duplicates(self.df)

        print("Step 3: Handling missing values...")
        self.df = self.handle_missing_values(self.df)

        print("Step 4: Converting strings and encoding categorical variables...")
        self.df = self.convert_strings(self.df)

        print("Step 5: Handling outliers...")
        self.df = self.handle_outliers(self.df, columns=['Price', 'Living Area'], method='IQR', remove=True)

        print("Step 6: Visualizing distributions and correlations...")
        self.visualize_distributions_and_correlation()

        print("Step 7: Saving processed data to CSV...")
        self.save_to_csv()

        end_time = time.time()
        print(f"--- Data Preprocessing Completed in {end_time - start_time:.2f} seconds ---")

    def load_data(self):
        """
        Loads data from the CSV file.
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully from {self.data_path}")
        print(f"Initial data shape: {self.df.shape}")

    @staticmethod
    def handle_duplicates(df):
        """
        Removes duplicate rows in the dataframe.
        Args:
            df (pd.DataFrame): The input dataframe.
        Returns:
            pd.DataFrame: The dataframe with duplicates removed.
        """
        duplicate_count = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicate_count}")
        if duplicate_count > 0:
            df = df.drop_duplicates(keep='first')
            print(f"Shape after removing duplicates: {df.shape}")
        return df

    @staticmethod
    def handle_missing_values(df):
        """
        Handles missing values in the dataframe.
        Args:
            df (pd.DataFrame): The input dataframe.
        Returns:
            pd.DataFrame: The dataframe with missing values handled.
        """
        print("Handling missing values...")
        columns_to_drop = ['id', 'Type of property', 'Type of sale', 'Garden', 'Garden area', 
                           'Surface of the land', 'Surface area of the plot of land', 
                           'Terrace', 'Number of rooms']
        print(f"Dropping columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop, axis=1)

        categorical_columns = ['Fully equipped kitchen', 'Furnished', 'State of the building']
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

        numerical_columns = ['Living Area', 'Terrace area', 'Number of facades', 'Swimming pool']
        numerical_imputer = SimpleImputer(strategy='median')
        df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

        print("Missing values handled successfully.")
        return df

    @staticmethod
    def convert_strings(df):
        """
        Encodes and processes string columns in the dataframe.
        Args:
            df (pd.DataFrame): The input dataframe.
        Returns:
            pd.DataFrame: The dataframe with encoded string columns.
        """
        counts = df['Subtype of property'].value_counts()
        mask = df['Subtype of property'].isin(counts[counts < 50].index)
        df.loc[mask, 'Subtype of property'] = 'other property'

        df = pd.get_dummies(df, columns=['Subtype of property', 'State of the building', 
                                         'Compound Listing', 'Fully equipped kitchen', 'Furnished'], 
                            prefix='d', drop_first=True, dtype=int)
        print("String columns converted to numerical representations.")
        return df

    @staticmethod
    def handle_outliers(df, columns, method='IQR', remove=True):
        """
        Handles outliers in specified columns using the IQR method.
        Args:
            df (pd.DataFrame): The input dataframe.
            columns (list): List of columns to check for outliers.
            method (str): The method for outlier handling.
            remove (bool): Whether to remove or cap outliers.
        Returns:
            pd.DataFrame: The dataframe with outliers handled.
        """
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if remove:
                rows_before = df.shape[0]
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                rows_after = df.shape[0]
                print(f"Outliers removed from {column}: {rows_before - rows_after} rows dropped.")
            else:
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                print(f"Outliers in {column} clipped.")

        print(f"Outliers handled for columns: {columns}")
        return df

    def visualize_distributions_and_correlation(self):
        """
        Visualizes distributions of key numerical variables and Spearman correlation heatmap.
        """
        print("Visualizing distributions and correlations...")
        num_vars = ['Price', 'Living Area']

        # Distribution plots
        for var in num_vars:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df[var], bins=30, kde=True, color='blue', alpha=0.6)
            plt.title(f"Distribution of {var} After Outlier Handling")
            plt.xlabel(var)
            plt.ylabel("Frequency")
            save_path = os.path.join(self.output_dir, f"{var}_distribution.png")
            plt.savefig(save_path)
            print(f"Saved distribution plot for {var} to {save_path}")
            plt.close()

        # Spearman correlation heatmap
        plt.figure(figsize=(10, 8))
        corr = self.df.corr(method='spearman')
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Spearman Correlation Heatmap")
        heatmap_path = os.path.join(self.output_dir, "spearman_correlation_heatmap.png")
        plt.savefig(heatmap_path)
        print(f"Spearman correlation heatmap saved to {heatmap_path}")
        plt.close()

    def save_to_csv(self):
        """
        Saves the processed dataframe to a uniquely named CSV file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"processed_data_{timestamp}.csv")
        self.df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")


# --- Main Function ---
def main():
    # Input parameters
    input_file = "Immoweb_scraping_result.csv" 
    output_dir = "."   # Directory to save processed data and visualizations

    # Initialize and run the preprocessing pipeline
    preprocessing = Preprocessing(
        data_path=input_file,
        output_dir=output_dir
    )

    preprocessing.run_pipeline()


if __name__ == "__main__":
    main()
