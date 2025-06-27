"""
Typer CLI for training recommendation models:

1. Load parquet data from ./data
2. Train candidate generator via CandidateGenerator.fit()
3. Build feature frame (candidate_score, watch_time label)
4. Train ranker via Ranker.fit()
5. Save trained models to data directory
6. Print Recall@100 and NDCG@10 on hold-out users

Note: This script is for general recommendation scenarios.
For HOMETOWN-specific recommendations, use model_trainer.py instead.
"""
import typer
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

# Import our models and services
from src.stream_rec.models.candidate_generator import CandidateGenerator
from src.stream_rec.models.ranker import Ranker
from src.stream_rec.services.feature_store import FeatureStore

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


def build_ranking_features(train_df: pd.DataFrame, candidate_generator: CandidateGenerator, 
                          feature_store: FeatureStore) -> pd.DataFrame:
    """Build feature dataframe for ranker training."""
    typer.echo("Building ranking features...")
    
    features_list = []
    unique_users = train_df['user_id'].unique()[:1000]  # Limit for demo
    
    for user_id in tqdm(unique_users, desc="Extracting features"):
        # Get user's actual interactions
        user_interactions = train_df[train_df['user_id'] == user_id]
        
        # Get candidate recommendations
        candidates = candidate_generator.top_k(user_id, k=100)
        
        # Get user and stream features
        user_features = feature_store.get_user_features(user_id)
        
        for candidate in candidates:
            stream_id = candidate['stream_id']
            stream_features = feature_store.get_stream_features(stream_id)
            
            # Check if user actually watched this stream
            actual_interaction = user_interactions[user_interactions['stream_id'] == stream_id]
            
            if len(actual_interaction) > 0:
                # Positive example
                label = 1
                watch_time = actual_interaction['watch_time'].iloc[0]
            else:
                # Negative example (not watched)
                label = 0
                watch_time = 0
            
            # Combine features
            feature_row = {
                'user_id': user_id,
                'stream_id': stream_id,
                'label': label,
                'watch_time': watch_time,
                'candidate_score': candidate['score'],
                # User features
                'user_engagement_rate': user_features.get('engagement_rate', 0.5),
                'user_session_length': user_features.get('avg_session_length', 60),
                'user_watch_hours': user_features.get('total_watch_hours', 100),
                # Stream features
                'stream_viewer_count': stream_features.get('viewer_count', 1000),
                'stream_duration': stream_features.get('duration_minutes', 120),
                'creator_followers': stream_features.get('creator_followers', 10000),
                'chat_activity': stream_features.get('chat_activity_rate', 1.0),
                # Interaction features
                'category_match': 1 if stream_features.get('category_id') in user_features.get('preferred_categories', []) else 0,
                'language_match': 1 if stream_features.get('language') == user_features.get('locale', 'en')[:2] else 0,
            }
            
            features_list.append(feature_row)
    
    return pd.DataFrame(features_list)


def evaluate_models(test_df: pd.DataFrame, candidate_generator: CandidateGenerator, 
                   ranker: Ranker) -> dict:
    """Evaluate trained models on test set."""
    typer.echo("Evaluating models...")
    
    recall_100_scores = []
    ndcg_10_scores = []
    
    test_users = test_df['user_id'].unique()[:100]  # Limit for demo
    
    for user_id in tqdm(test_users, desc="Evaluating"):
        # Get user's test interactions
        user_test = test_df[test_df['user_id'] == user_id]
        actual_streams = set(user_test['stream_id'].tolist())
        
        if len(actual_streams) == 0:
            continue
        
        # Get recommendations
        candidates = candidate_generator.top_k(user_id, k=200)
        ranked_recs = ranker.rank(user_id, candidates, top_n=100)
        
        # Calculate Recall@100
        recommended_streams = set([rec['stream_id'] for rec in ranked_recs])
        recall_100 = len(actual_streams.intersection(recommended_streams)) / len(actual_streams)
        recall_100_scores.append(recall_100)
        
        # Calculate NDCG@10
        top_10_recs = ranked_recs[:10]
        dcg = 0
        for i, rec in enumerate(top_10_recs):
            if rec['stream_id'] in actual_streams:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG (assume all relevant items at top)
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(actual_streams))))
        
        ndcg_10 = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_10_scores.append(ndcg_10)
    
    return {
        'recall_100': np.mean(recall_100_scores),
        'ndcg_10': np.mean(ndcg_10_scores)
    }


@app.command()
def train(
    data_dir: str = typer.Option("./data", help="Directory containing parquet files"),
    test_ratio: float = typer.Option(0.2, help="Fraction of users for testing"),
    force_synthetic: bool = typer.Option(False, help="Force generation of synthetic data")
):
    """Train recommendation models and evaluate performance."""
    typer.echo("🚀 Starting model training pipeline...")
    
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
    
    # Initialize components
    typer.echo("Initializing models and feature store...")
    candidate_generator = CandidateGenerator()
    ranker = Ranker()
    feature_store = FeatureStore()
    
    # Train CandidateGenerator model
    typer.echo("🔥 Training candidate generator model...")
    candidate_generator.fit(train_df)
    
    # Build ranking features
    ranking_features = build_ranking_features(train_df, candidate_generator, feature_store)
    
    # Train Ranker (XGBoost)
    typer.echo("🎯 Training XGBoost ranker...")
    ranker.fit(ranking_features)
    
    # Evaluate models
    metrics = evaluate_models(test_df, candidate_generator, ranker)
    
    # Print results
    typer.echo("\n📊 Evaluation Results:")
    typer.echo(f"Recall@100: {metrics['recall_100']:.4f}")
    typer.echo(f"NDCG@10: {metrics['ndcg_10']:.4f}")
    
    typer.echo("\n✅ Training completed successfully!")
    typer.echo(f"Models saved to:")
    typer.echo(f"  - Candidate Generator: data/candidate_generator.ckpt")
    typer.echo(f"  - Ranker: data/ranker.pkl")


if __name__ == "__main__":
    app()
