import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from config import DB_PATH, MAX_RETRIES
import os
from typing import Tuple

logger = logging.getLogger(__name__)

class PredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(os.path.dirname(DB_PATH), 'prediction_model.joblib')
        self.scaler_path = os.path.join(os.path.dirname(DB_PATH), 'scaler.joblib')
        self.load_model()

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Expanding grid search to include GradientBoosting
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, self.model.predict_proba(X_test_scaled), multi_class='ovr')

        logger.info(f"Model performance: Accuracy={accuracy:.2f}, Precision={precision:.2f}, "
                    f"Recall={recall:.2f}, F1-score={f1:.2f}, ROC AUC={roc_auc:.2f}")

        self.save_model()

    def predict(self, X: pd.DataFrame) -> Tuple[str, float]:
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[0]
        predicted_class = self.model.classes_[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        return predicted_class, confidence

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return feature_importance

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Scaler saved to {self.scaler_path}")

    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Model loaded from {self.model_path}")
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.info("No existing model found. A new model will be trained.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, self.model.predict_proba(X_scaled), multi_class='ovr')
        }
