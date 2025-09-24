import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion


class DataTransformationConfig:
    def __init__(self):
        self.train_array_path = os.path.join("artifacts", "train.npy")
        self.test_array_path = os.path.join("artifacts", "test.npy")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("Starting data transformation process...")

        try:
            # 1. Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # 2. Separate input features (X) and target (y)
            X_train = train_df.drop("label", axis=1)
            y_train = train_df["label"]

            X_test = test_df.drop("label", axis=1)
            y_test = test_df["label"]

            # 3. Encode target labels
            logger.info("Encoding labels...")
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)

            # 4. Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 5. Combine features + target
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]

            # 6. Save arrays
            np.save(self.config.train_array_path, train_arr)
            np.save(self.config.test_array_path, test_arr)

            logger.info("Data transformation completed successfully ✅")
            return (
                self.config.train_array_path,
                self.config.test_array_path,
                self.label_encoder  # 🔥 return encoder too
            )

        except Exception as e:
            logger.error("Error occurred during data transformation ❌")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("notebook/data/crop_recommendation.csv")

    transformation = DataTransformation()
    train_arr, test_arr, encoder = transformation.initiate_data_transformation(train_path, test_path)

    print(f"✅ Transformed train array: {train_arr}")
    print(f"✅ Transformed test array: {test_arr}")
