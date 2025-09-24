import os
import sys
import pickle
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging

# ----------------- Logger Setup -----------------
LOG_FILE = os.path.join("logs", "running_logs.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    filemode='a'
)

logger = logging.getLogger(__name__)

# Also print logs to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)


# ----------------- Exception Class -----------------
class CustomException(Exception):
    def __init__(self, message, errors=sys):
        super().__init__(message)
        self.errors = errors


# ----------------- Model Trainer -----------------
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

    def save_model(self, model, label_encoder=None):
        """Save the trained model and label encoder"""
        try:
            with open(self.config.model_path, "wb") as f:
                pickle.dump({"model": model, "encoder": label_encoder}, f)  # 🔥 save dict
            logger.info(f"Model + encoder saved at {self.config.model_path}")
        except Exception as e:
            raise CustomException(f"Error saving model: {e}", sys)

    def initiate_model_trainer(self, X_train, y_train, label_encoder=None):
        try:
            logger.info("Starting model training...")
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            logger.info("Model training completed successfully")

            self.save_model(model, label_encoder)
            return model

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model...")
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"Model Accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)


# ----------------- Test Run -----------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    trainer = ModelTrainer()
    model = trainer.initiate_model_trainer(X_train, y_train)
    trainer.evaluate_model(model, X_test, y_test)
