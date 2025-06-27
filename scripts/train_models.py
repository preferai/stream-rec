"""
Typer CLI for training HOMETOWN recommendation models:

1. Load parquet data from ./data (or generate synthetic data)
2. Check for required hometown_train.parquet file
3. Train HOMETOWN ML model via HometownModelTrainer
4. Initialize DataStore and evaluate both basic and ML algorithms
5. Save trained model to models/hometown_model.pkl
6. Print training accuracy, AUC score, and algorithm comparison

This script now uses the working HometownModelTrainer implementation
instead of the previously missing CandidateGenerator/Ranker classes.
"""
import typer
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import sys

# Add src to Python path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import our implemented services
from stream_rec.services.model_trainer import HometownModelTrainer
from stream_rec.services.data_store import DataStore
from stream_rec.services.hometown_recommender import HometownRecommender

app = typer.Typer()


def load_parquet_data(data_dir: Path) -> pd.DataFrame:
    """Load interaction data from parquet files."""
    data_files = list(data_dir.glob("*.parquet"))
    
    if not data_files:
        typer.echo(f"No parquet files found in {data_dir}")
        typer.echo("Generating synthetic data for demo...")
        return generate_synthetic_data()
    
    typer.echo(f"Loading {len(data_files)} parquet files...")
    
    dfs = []
    for file_path in data_files:
        df = pd.read_parquet(file_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    typer.echo(f"Loaded {len(combined_df)} interactions")
    
    return combined_df


def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic interaction data for demo purposes."""
    num_users = 10000
    num_streams = 50000
    num_interactions = 500000
    
    # Generate realistic interaction patterns
    data = []
    
    # Popular streams get more interactions (power law distribution)
    stream_popularity = np.random.zipf(1.5, num_streams)
    stream_ids = [f"stream_{i:06d}" for i in range(num_streams)]
    
    # Active users watch more streams
    user_activity = np.random.gamma(2, 2, num_users)
    user_ids = [f"user_{i:06d}" for i in range(num_users)]
    
    for _ in range(num_interactions):
        # Select user (bias toward active users)
        user_idx = np.random.choice(num_users, p=user_activity/user_activity.sum())
        user_id = user_ids[user_idx]
        
        # Select stream (bias toward popular streams)
        stream_idx = np.random.choice(num_streams, p=stream_popularity/stream_popularity.sum())
        stream_id = stream_ids[stream_idx]
        
        # Generate watch time (longer for preferred content)
        base_watch_time = np.random.exponential(30)  # minutes
        watch_time = max(1, base_watch_time)
        
        # Rating based on watch time (longer = higher rating)
        if watch_time > 60:
            rating = np.random.choice([4, 5], p=[0.3, 0.7])
        elif watch_time > 20:
            rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        else:
            rating = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        
        data.append({
            'user_id': user_id,
            'stream_id': stream_id,
            'watch_time': watch_time,
            'rating': rating,
            'timestamp': datetime.now().timestamp() - random.randint(0, 30*24*3600)  # Last 30 days
        })
    
    return pd.DataFrame(data)


def train_test_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by users for evaluation."""
    unique_users = df['user_id'].unique()
    
    # Ensure each user has enough interactions
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index.tolist()
    
    test_users = set(random.sample(valid_users, int(len(valid_users) * test_ratio)))
    
    train_df = df[~df['user_id'].isin(test_users)]
    test_df = df[df['user_id'].isin(test_users)]
    
    typer.echo(f"Train: {len(train_df)} interactions from {train_df['user_id'].nunique()} users")
    typer.echo(f"Test: {len(test_df)} interactions from {test_df['user_id'].nunique()} users")
    
    return train_df, test_df


def train_hometown_model(data_dir: Path) -> dict:
    """Train the HOMETOWN model using HometownModelTrainer."""
    typer.echo("🏠 Training HOMETOWN model...")
    
    trainer = HometownModelTrainer(data_dir=str(data_dir))
    
    # Train the model
    results = trainer.train()
    
    # Save the model
    trainer.save_model()
    
    return results


def evaluate_hometown_model(test_df: pd.DataFrame, data_store: DataStore) -> dict:
    """Evaluate the HOMETOWN model with basic metrics."""
    typer.echo("📊 Evaluating HOMETOWN model...")
    
    # Initialize recommenders
    basic_recommender = HometownRecommender(data_store)
    ml_recommender = HometownRecommender(data_store, model_path="models/hometown_model.pkl")
    
    # Simple evaluation on a subset of users
    test_users = test_df['user_id'].unique()[:50]  # Limit for demo
    
    basic_scores = []
    ml_scores = []
    
    for user_id in tqdm(test_users, desc="Evaluating"):
        try:
            # Get recommendations from both algorithms
            basic_recs = basic_recommender.recommend_streams(user_id, max_results=10)
            ml_recs = ml_recommender.recommend_streams(user_id, max_results=10)
            
            # Simple scoring based on average scores
            if basic_recs:
                basic_avg_score = sum(rec['score'] for rec in basic_recs) / len(basic_recs)
                basic_scores.append(basic_avg_score)
            
            if ml_recs:
                ml_avg_score = sum(rec['score'] for rec in ml_recs) / len(ml_recs)
                ml_scores.append(ml_avg_score)
                
        except Exception as e:
            typer.echo(f"Warning: Failed to evaluate user {user_id}: {e}")
            continue
    
    return {
        'basic_avg_score': np.mean(basic_scores) if basic_scores else 0,
        'ml_avg_score': np.mean(ml_scores) if ml_scores else 0,
        'users_evaluated': len(basic_scores)
    }


@app.command()
def train(
    data_dir: str = typer.Option("./data", help="Directory containing parquet files"),
    test_ratio: float = typer.Option(0.2, help="Fraction of users for testing"),
    force_synthetic: bool = typer.Option(False, help="Force generation of synthetic data")
):
    """Train HOMETOWN recommendation model and evaluate performance."""
    typer.echo("🚀 Starting HOMETOWN model training pipeline...")
    
    # Setup directories
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Load data
    if force_synthetic:
        df = generate_synthetic_data()
    else:
        df = load_parquet_data(data_path)
    
    # Train/test split
    train_df, test_df = train_test_split(df, test_ratio)
    
    # Check if we have HOMETOWN training data
    hometown_train_path = data_path / "hometown_train.parquet"
    if not hometown_train_path.exists():
        typer.echo("⚠️  No HOMETOWN training data found.")
        typer.echo("   Run: uv run python scripts/generate_hometown_dataset.py")
        typer.echo("   This will generate the required hometown_train.parquet file.")
        return
    
    # Train HOMETOWN model
    try:
        results = train_hometown_model(data_path)
        
        # Initialize data store for evaluation
        typer.echo("� Initializing data store for evaluation...")
        data_store = DataStore()
        
        # Evaluate model
        eval_metrics = evaluate_hometown_model(test_df, data_store)
        
        # Print results
        typer.echo("\n📊 Training Results:")
        typer.echo(f"Train Accuracy: {results['train_accuracy']:.4f}")
        typer.echo(f"Test Accuracy: {results['test_accuracy']:.4f}")
        typer.echo(f"AUC Score: {results['auc_score']:.4f}")
        
        typer.echo("\n📊 Evaluation Results:")
        typer.echo(f"Basic Algorithm Avg Score: {eval_metrics['basic_avg_score']:.4f}")
        typer.echo(f"ML Algorithm Avg Score: {eval_metrics['ml_avg_score']:.4f}")
        typer.echo(f"Users Evaluated: {eval_metrics['users_evaluated']}")
        
        typer.echo("\n✅ HOMETOWN model training completed successfully!")
        typer.echo(f"Model saved to: models/hometown_model.pkl")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    app()
