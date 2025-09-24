import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self, input_file: str):
        logger.info("Starting data ingestion process...")

        try:
            # 1. Read raw dataset
            logger.info(f"Reading dataset from {input_file}")
            df = pd.read_csv(input_file)
            logger.info(f"Dataset shape: {df.shape}")

            # 2. Create artifacts directory if not exists
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved at {self.config.raw_data_path}")

            # 3. Train-test split
            logger.info("Performing train-test split (80%-20%)")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train & test sets
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info(f"Train data saved at {self.config.train_data_path}")
            logger.info(f"Test data saved at {self.config.test_data_path}")
            logger.info("Data ingestion completed successfully ")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("notebook/data/crop_recommendation.csv")
    print(f"Train data: {train_path}, Test data: {test_path}")
