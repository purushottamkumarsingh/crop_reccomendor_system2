import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger


def main():
    logger.info("🚀 Starting Crop Recommendation Training Pipeline...")

    # 1. Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion("notebook/data/crop_recommendation.csv")

    # 2. Data Transformation
    transformation = DataTransformation()
    train_array_path, test_array_path, encoder = transformation.initiate_data_transformation(train_path, test_path)

    train_arr = np.load(train_array_path)
    test_arr = np.load(test_array_path)

    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

    # 3. Model Training
    trainer = ModelTrainer()
    model = trainer.initiate_model_trainer(X_train, y_train, label_encoder=encoder)  # 🔥 encoder passed here
    acc = trainer.evaluate_model(model, X_test, y_test)

    logger.info(f"✅ Final Model Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
