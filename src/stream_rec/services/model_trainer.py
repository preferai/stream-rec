"""
Optional ML model trainer for enhanced HOMETOWN recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path


class HometownModelTrainer:
    """
    Trains an ML model to predict click probability for HOMETOWN scenario.
    
    This enhances the basic proximity boost algorithm with learned patterns
    from historical user-stream interactions.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.feature_columns = [
            'distance_km', 'proximity_boost', 'is_local_match', 'is_regional_match',
            'same_city', 'same_language', 'user_local_preference', 'user_total_watch_hours',
            'stream_quality_score', 'stream_avg_viewers', 'creator_followers',
            'is_partnered', 'mature_content', 'category_match'
        ]
    
    def load_training_data(self) -> pd.DataFrame:
        """Load the training dataset."""
        train_path = self.data_dir / "hometown_train.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        return pd.read_parquet(train_path)
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for training."""
        # Ensure all required features exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        X = df[self.feature_columns].copy()
        y = df['clicked'].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        return X, y
    
    def train(self, test_size: float = 0.2) -> dict:
        """
        Train the HOMETOWN recommendation model.
        
        Returns:
            dict: Training metrics and model performance
        """
        print("🏠 Training HOMETOWN ML Model...")
        
        # Load data
        df = self.load_training_data()
        print(f"📊 Loaded {len(df):,} training samples")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"🔧 Training on {len(X_train):,} samples, testing on {len(X_test):,}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.coef_[0]))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
        
        results = {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "auc_score": auc_score,
            "feature_importance": feature_importance,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"✅ Model trained successfully!")
        print(f"   • Train Accuracy: {train_score:.3f}")
        print(f"   • Test Accuracy: {test_score:.3f}")
        print(f"   • AUC Score: {auc_score:.3f}")
        
        print(f"\n🎯 Top Feature Importances:")
        for feature, importance in list(feature_importance.items())[:5]:
            print(f"   • {feature:<25}: {importance:>6.3f}")
        
        return results
    
    def save_model(self, model_path: str = "models/hometown_model.pkl"):
        """Save the trained model."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self, model_path: str = "models/hometown_model.pkl"):
        """Load a pre-trained model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"📁 Model loaded from {model_path}")


def main():
    """Train and save the HOMETOWN model."""
    trainer = HometownModelTrainer()
    
    try:
        # Train model
        results = trainer.train()
        
        # Save model
        trainer.save_model()
        
        print("\n🎉 HOMETOWN model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    main()
