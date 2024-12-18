import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os

class XGBRegressionPipeline:
    """
    A pipeline for building and evaluating an XGBoost regression model. It includes data loading, preprocessing,
    cross-validation, hyperparameter tuning, model training, and evaluation.

    Parameters:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target variable in the dataset.
        categorical_features (list): List of categorical feature names in the dataset.
    """
    def __init__(self, data_path, target_column, categorical_features):
        self.data_path = data_path
        self.target_column = target_column
        self.categorical_features = categorical_features
        self.model = None  # Placeholder for the model that will be updated with GridSearchCV

    def load_data(self):
        """
        Loads the dataset from the specified path and separates it into features and target.
        Returns:
            tuple: Features (X) and target (y).
        """
        start_time = time.time()
        self.df = pd.read_csv(self.data_path)
        self.y = self.df[self.target_column]
        self.X = self.df.drop(self.target_column, axis=1)
        end_time = time.time()
        print(f"Data loading time: {end_time - start_time:.2f} seconds")
        return self.X, self.y

    def split_data(self, test_size=0.2):
        """
        Splits the dataset into training and testing sets.
        Parameters:
            test_size (float): Proportion of the data to use as the test set.
        """
        start_time = time.time()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        end_time = time.time()
        print(f"Data splitting time: {end_time - start_time:.2f} seconds")

    def preprocess_data(self):
        """
        Preprocesses the data by applying log transformation to numerical features
        and normalizing all features to a [0, 1] range.
        """
        start_time = time.time()
        numeric_feats = [col for col in self.X_train.columns if col not in self.categorical_features]
        self.X_train[numeric_feats] = np.log1p(self.X_train[numeric_feats])
        self.X_test[numeric_feats] = np.log1p(self.X_test[numeric_feats])

        self.scaler = MinMaxScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=self.X_test.columns
        )
        end_time = time.time()
        print(f"Data preprocessing time: {end_time - start_time:.2f} seconds")

    def perform_cross_validation(self):
        """
        Performs K-Fold cross-validation to evaluate the model's performance.
        """
        start_time = time.time()
        print("\nPerforming Cross-Validation...")
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        results = cross_val_score(self.model, self.X, self.y, cv=kfold)
        end_time = time.time()
        print(f"Cross-Validation Accuracy: {results.mean() * 100:.2f}% ({results.std() * 100:.2f}%)")
        print(f"Cross-validation time: {end_time - start_time:.2f} seconds")

    def perform_hyperparameter_tuning(self):
        """
        Performs GridSearchCV to find the best hyperparameters for the XGBoost model.
        """
        start_time = time.time()
        print("\nPerforming Hyperparameter Tuning...")
        params = {
            'objective': ['reg:squarederror'], 
            'max_depth': [6, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [300, 400],
            'subsample': [0.4, 0.8],
        }
        grid_mse = GridSearchCV(
            estimator=xgb.XGBRegressor(), 
            param_grid=params, 
            scoring='neg_mean_squared_error', 
            cv=4, 
            verbose=1
        )
        grid_mse.fit(self.X, self.y)
        end_time = time.time()
        print(f"Best Parameters: {grid_mse.best_params_}")
        print(f"Best RMSE: {np.sqrt(np.abs(grid_mse.best_score_)):.4f}")
        print(f"Hyperparameter tuning time: {end_time - start_time:.2f} seconds")
        self.model = grid_mse.best_estimator_

    def train_model(self):
        """
        Trains the XGBoost model on the training dataset.
        """
        start_time = time.time()
        if not self.model:
            raise ValueError("Model has not been initialized. Perform hyperparameter tuning first.")
        self.model.fit(self.X_train_scaled, self.y_train)
        end_time = time.time()
        print(f"Model training time: {end_time - start_time:.2f} seconds")

    def evaluate_model(self):
        """
        Evaluates the model using various metrics: MAE, RMSE, R², MAPE, and sMAPE.
        Returns:
            tuple: Dictionary of metrics, training predictions, testing predictions.
        """
        start_time = time.time()
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        end_time = time.time()

        metrics = {
            "train": {
                "MAE": mean_absolute_error(self.y_train, y_train_pred),
                "RMSE": np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                "R2": r2_score(self.y_train, y_train_pred) * 100,
                "MAPE": np.mean(np.abs((self.y_train - y_train_pred) / self.y_train)) * 100,
                "sMAPE": np.mean(200 * np.abs(self.y_train - y_train_pred) / 
                                 (np.abs(self.y_train) + np.abs(y_train_pred))) * 100
            },
            "test": {
                "MAE": mean_absolute_error(self.y_test, y_test_pred),
                "RMSE": np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                "R2": r2_score(self.y_test, y_test_pred) * 100,
                "MAPE": np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100,
                "sMAPE": np.mean(200 * np.abs(self.y_test - y_test_pred) / 
                                  (np.abs(self.y_test) + np.abs(y_test_pred))) * 100
            }
        }

        print(f"Model evaluation time: {end_time - start_time:.2f} seconds")
        return metrics, y_train_pred, y_test_pred

    def plot_feature_importance(self):
        """
        Plots and saves the feature importance graph.
        """
        save_path = "feature_importance.png"
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(self.model, importance_type='weight', max_num_features=15)
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}.")
        plt.show()

    def plot_predictions(self, y_test_pred):
        """
        Plots and saves the actual vs. predicted values scatter plot.
        """
        save_path = "actual_vs_predicted.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_test_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Values")
        plt.savefig(save_path)
        print(f"Prediction scatter plot saved to {save_path}.")
        plt.show()

    def run(self):
        """
        Executes the entire pipeline: data loading, preprocessing, training, evaluation, and plotting.
        """
        start_time = time.time()
        self.load_data()
        self.split_data()
        self.preprocess_data()
        self.perform_hyperparameter_tuning()
        self.perform_cross_validation()
        self.train_model()
        metrics, _, y_test_pred = self.evaluate_model()
        self.plot_feature_importance()
        self.plot_predictions(y_test_pred)
        end_time = time.time()
        print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")

        print("\nTraining Metrics:")
        for metric, value in metrics["train"].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nTesting Metrics:")
        for metric, value in metrics["test"].items():
            print(f"{metric}: {value:.4f}")

# Usage example
if __name__ == "__main__":
    categorical_features = ['Locality', 'Fully equipped kitchen', 'Fireplace', 'Swimming pool', 'Furnished',
                            'd_APARTMENT_BLOCK', 'd_DUPLEX', 'd_EXCEPTIONAL_PROPERTY', 'd_FLAT_STUDIO', 'd_GROUND_FLOOR',
                            'd_HOUSE', 'd_MANSION', 'd_MIXED_USE_BUILDING', 'd_PENTHOUSE', 'd_TOWN_HOUSE', 'd_VILLA',
                            'd_other property', 'd_GOOD', 'd_JUST_RENOVATED', 'd_TO_BE_DONE_UP', 'd_TO_RENOVATE', 
                            'd_TO_RESTORE', 'd_UNKNOWN', 'd_single']
    
    pipeline = XGBRegressionPipeline(
        data_path='Data/processed_data_20241209_134057.csv',
        target_column='Price',
        categorical_features=categorical_features
    )
    pipeline.run()
