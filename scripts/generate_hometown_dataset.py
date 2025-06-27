#!/usr/bin/env python3
"""
Generate Synthetic Dataset for HOMETOWN Scenario

This script generates realistic synthetic data for training and testing the HOMETOWN recommendation scenario.
The HOMETOWN scenario orders the front-page grid so that live channels from the user's own city/region appear first.

Key KPI: CTR on location-matched stream impressions
Implementation: Proximity boost B = 1 / (1 + distance_km) for local channels

Data generated:
1. User profiles with geo-locations (cities, coordinates)
2. Stream metadata with streamer locations
3. Historical user-stream interactions with location-based preferences
4. Click-through events with location bias
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
from geopy.distance import geodesic
from pathlib import Path
import typer

app = typer.Typer()

# Global constants for realistic data generation
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Major cities worldwide with coordinates (lat, lon)
CITIES_DATA = {
    # North America
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Toronto': (43.6532, -79.3832),
    'Vancouver': (49.2827, -123.1207),
    'Mexico City': (19.4326, -99.1332),
    
    # Europe
    'London': (51.5074, -0.1278),
    'Paris': (48.8566, 2.3522),
    'Berlin': (52.5200, 13.4050),
    'Madrid': (40.4168, -3.7038),
    'Rome': (41.9028, 12.4964),
    'Amsterdam': (52.3676, 4.9041),
    'Stockholm': (59.3293, 18.0686),
    'Prague': (50.0755, 14.4378),
    'Vienna': (48.2082, 16.3738),
    'Warsaw': (52.2297, 21.0122),
    
    # Asia-Pacific
    'Tokyo': (35.6762, 139.6503),
    'Seoul': (37.5665, 126.9780),
    'Beijing': (39.9042, 116.4074),
    'Shanghai': (31.2304, 121.4737),
    'Mumbai': (19.0760, 72.8777),
    'Bangkok': (13.7563, 100.5018),
    'Singapore': (1.3521, 103.8198),
    'Sydney': (33.8688, 151.2093),
    'Melbourne': (37.8136, 144.9631),
    
    # South America
    'São Paulo': (-23.5505, -46.6333),
    'Buenos Aires': (-34.6118, -58.3960),
    'Lima': (-12.0464, -77.0428),
    
    # Africa & Middle East
    'Dubai': (25.2048, 55.2708),
    'Tel Aviv': (32.0853, 34.7818),
    'Cairo': (30.0444, 31.2357),
    'Cape Town': (-33.9249, 18.4241)
}

CITIES = list(CITIES_DATA.keys())

# Stream categories (games/genres)
STREAM_CATEGORIES = [
    'valorant', 'league_of_legends', 'fortnite', 'apex_legends', 'cs2',
    'minecraft', 'gta_v', 'wow', 'dota2', 'overwatch2',
    'just_chatting', 'music', 'art', 'cooking', 'irl',
    'sports', 'chess', 'programming', 'educational', 'creative'
]

# Languages by region
LANGUAGE_BY_REGION = {
    'en': ['New York', 'Los Angeles', 'Chicago', 'Toronto', 'Vancouver', 'London', 'Sydney', 'Melbourne', 'Cape Town'],
    'es': ['Mexico City', 'Madrid', 'Buenos Aires', 'Lima'],
    'fr': ['Paris'],
    'de': ['Berlin', 'Vienna'],
    'it': ['Rome'],
    'nl': ['Amsterdam'],
    'sv': ['Stockholm'],
    'cs': ['Prague'],
    'pl': ['Warsaw'],
    'ja': ['Tokyo'],
    'ko': ['Seoul'],
    'zh': ['Beijing', 'Shanghai'],
    'hi': ['Mumbai'],
    'th': ['Bangkok'],
    'pt': ['São Paulo'],
    'ar': ['Dubai', 'Cairo'],
    'he': ['Tel Aviv']
}

def get_city_language(city: str) -> str:
    """Get the primary language for a city."""
    for lang, cities in LANGUAGE_BY_REGION.items():
        if city in cities:
            return lang
    return 'en'  # Default to English

def calculate_distance_km(city1: str, city2: str) -> float:
    """Calculate distance between two cities in kilometers."""
    if city1 == city2:
        return 0.0
    
    coord1 = CITIES_DATA[city1]
    coord2 = CITIES_DATA[city2]
    return geodesic(coord1, coord2).kilometers

def generate_user_profiles(num_users: int) -> pd.DataFrame:
    """Generate user profiles with realistic location distributions."""
    users = []
    
    # City distribution (some cities more popular than others)
    city_weights = {
        'New York': 0.08, 'Los Angeles': 0.06, 'Chicago': 0.04, 'Toronto': 0.03,
        'London': 0.07, 'Paris': 0.04, 'Berlin': 0.03, 'Madrid': 0.02,
        'Tokyo': 0.06, 'Seoul': 0.03, 'Beijing': 0.03, 'Shanghai': 0.03,
        'Mumbai': 0.04, 'Bangkok': 0.02, 'Singapore': 0.02, 'Sydney': 0.02,
        'São Paulo': 0.03, 'Buenos Aires': 0.02, 'Dubai': 0.02, 'Tel Aviv': 0.015
    }
    
    # Normalize remaining cities
    remaining_weight = 1.0 - sum(city_weights.values())
    remaining_cities = [city for city in CITIES if city not in city_weights]
    for city in remaining_cities:
        city_weights[city] = remaining_weight / len(remaining_cities)
    
    cities_list = list(city_weights.keys())
    weights_list = list(city_weights.values())
    
    for i in range(num_users):
        city = np.random.choice(cities_list, p=weights_list)
        lat, lon = CITIES_DATA[city]
        
        # Add some noise to coordinates (within city bounds)
        lat_noise = np.random.normal(0, 0.05)  # ~5km variance
        lon_noise = np.random.normal(0, 0.05)
        
        user = {
            'user_id': f'user_{i:06d}',
            'city': city,
            'latitude': lat + lat_noise,
            'longitude': lon + lon_noise,
            'language': get_city_language(city),
            'timezone_offset': np.random.choice([-8, -5, 0, 1, 2, 8, 9], p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]),
            'registration_date': datetime.now() - timedelta(days=np.random.randint(1, 1095)),  # 0-3 years
            'total_watch_hours': np.random.gamma(50, 2),  # Gamma distribution for realistic watch time
            'preferred_categories': random.sample(STREAM_CATEGORIES, k=np.random.randint(2, 6)),
            'local_preference_strength': np.random.beta(2, 3)  # 0-1, bias toward lower values (some prefer local)
        }
        users.append(user)
    
    return pd.DataFrame(users)

def generate_stream_metadata(num_streams: int) -> pd.DataFrame:
    """Generate stream metadata with streamer locations."""
    streams = []
    
    # Streamers are more concentrated in certain tech/gaming hubs
    streamer_city_weights = {
        'Los Angeles': 0.12, 'New York': 0.08, 'Chicago': 0.05, 'Toronto': 0.04,
        'London': 0.10, 'Berlin': 0.06, 'Paris': 0.04, 'Stockholm': 0.03,
        'Tokyo': 0.08, 'Seoul': 0.07, 'Shanghai': 0.03,
        'Sydney': 0.03, 'Melbourne': 0.02,
        'São Paulo': 0.02, 'Buenos Aires': 0.015,
        'Dubai': 0.02, 'Tel Aviv': 0.02
    }
    
    # Normalize remaining cities
    remaining_weight = 1.0 - sum(streamer_city_weights.values())
    remaining_cities = [city for city in CITIES if city not in streamer_city_weights]
    for city in remaining_cities:
        streamer_city_weights[city] = remaining_weight / len(remaining_cities)
    
    cities_list = list(streamer_city_weights.keys())
    weights_list = list(streamer_city_weights.values())
    
    for i in range(num_streams):
        city = np.random.choice(cities_list, p=weights_list)
        lat, lon = CITIES_DATA[city]
        
        # Stream quality follows power law (few high-quality streams)
        quality_score = np.random.pareto(1.2)  # Pareto distribution
        quality_score = min(quality_score, 10.0)  # Cap at 10
        
        stream = {
            'stream_id': f'stream_{i:06d}',
            'creator_id': f'creator_{i:06d}',
            'city': city,
            'latitude': lat,
            'longitude': lon,
            'language': get_city_language(city),
            'category_id': np.random.choice(STREAM_CATEGORIES),
            'title': f'Stream Title {i}',
            'description': f'Stream description for stream {i}',
            'avg_viewer_count': max(1, int(np.random.exponential(100) * quality_score)),
            'creator_followers': max(10, int(np.random.exponential(1000) * quality_score)),
            'stream_quality_score': quality_score,
            'typical_duration_minutes': np.random.gamma(2, 60),  # 2-hour average
            'is_partnered': np.random.choice([True, False], p=[0.3, 0.7]),
            'mature_content': np.random.choice([True, False], p=[0.2, 0.8]),
            'tags': random.sample(['gaming', 'educational', 'entertainment', 'music', 'art', 'irl'], 
                                k=np.random.randint(1, 4))
        }
        streams.append(stream)
    
    return pd.DataFrame(streams)

def generate_interactions(users_df: pd.DataFrame, streams_df: pd.DataFrame, 
                         num_interactions: int) -> pd.DataFrame:
    """Generate user-stream interactions with location bias."""
    interactions = []
    
    # Create lookup dictionaries
    user_city_map = dict(zip(users_df['user_id'], users_df['city']))
    user_local_pref = dict(zip(users_df['user_id'], users_df['local_preference_strength']))
    stream_city_map = dict(zip(streams_df['stream_id'], streams_df['city']))
    
    for _ in range(num_interactions):
        # Select user (some users more active than others)
        user_id = np.random.choice(users_df['user_id'], 
                                 p=users_df['total_watch_hours'] / users_df['total_watch_hours'].sum())
        
        user_city = user_city_map[user_id]
        local_preference = user_local_pref[user_id]
        
        # Calculate location-based stream probabilities
        stream_probs = []
        for _, stream in streams_df.iterrows():
            stream_city = stream['city']
            base_prob = stream['stream_quality_score']  # Base attractiveness
            
            # Apply location boost
            distance_km = calculate_distance_km(user_city, stream_city)
            location_boost = 1 / (1 + distance_km / 1000)  # Normalize by 1000km
            
            # Apply user's local preference
            if distance_km < 100:  # Same city/region
                location_weight = 1 + (local_preference * 3)  # Up to 4x boost for local content
            elif distance_km < 1000:  # Same country/region
                location_weight = 1 + (local_preference * 1)  # Up to 2x boost
            else:
                location_weight = 1.0
            
            final_prob = base_prob * location_boost * location_weight
            stream_probs.append(final_prob)
        
        # Normalize probabilities
        stream_probs = np.array(stream_probs)
        stream_probs = stream_probs / stream_probs.sum()
        
        # Select stream
        stream_id = np.random.choice(streams_df['stream_id'], p=stream_probs)
        
        # Generate interaction details
        user_city = user_city_map[user_id]
        stream_city = stream_city_map[stream_id]
        distance_km = calculate_distance_km(user_city, stream_city)
        
        # Watch time influenced by location match
        base_watch_time = np.random.exponential(45)  # 45 minutes average
        if distance_km < 100:  # Local content
            watch_time_multiplier = 1 + (local_preference * 0.5)  # Up to 50% longer
        else:
            watch_time_multiplier = 1.0
        
        watch_time = max(1, base_watch_time * watch_time_multiplier)
        
        # Click-through (1 if watched, 0 if just impressed)
        # Local content has higher CTR
        base_ctr = 0.15  # 15% base CTR
        if distance_km < 100:
            ctr_boost = local_preference * 0.3  # Up to 30% absolute increase
        else:
            ctr_boost = 0.0
        
        clicked = np.random.choice([0, 1], p=[1 - (base_ctr + ctr_boost), base_ctr + ctr_boost])
        
        interaction = {
            'user_id': user_id,
            'stream_id': stream_id,
            'user_city': user_city,
            'stream_city': stream_city,
            'distance_km': distance_km,
            'clicked': clicked,
            'watch_time_minutes': watch_time if clicked else 0,
            'impression_timestamp': datetime.now() - timedelta(seconds=np.random.randint(0, 30*24*3600)),
            'is_local_match': distance_km < 100,  # Same city/region
            'is_regional_match': distance_km < 1000,  # Same country/region
            'session_id': f'session_{np.random.randint(1000000, 9999999)}',
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1]),
            'page_position': np.random.randint(1, 51)  # Position on page (1-50)
        }
        interactions.append(interaction)
    
    return pd.DataFrame(interactions)

def create_hometown_features(interactions_df: pd.DataFrame, users_df: pd.DataFrame, 
                           streams_df: pd.DataFrame) -> pd.DataFrame:
    """Create feature dataset specifically for HOMETOWN scenario training."""
    features = []
    
    # Group by user for efficiency
    for user_id in interactions_df['user_id'].unique():
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        user_data = users_df[users_df['user_id'] == user_id].iloc[0]
        
        for _, interaction in user_interactions.iterrows():
            stream_data = streams_df[streams_df['stream_id'] == interaction['stream_id']].iloc[0]
            
            # Calculate location-based features
            distance_km = interaction['distance_km']
            proximity_boost = 1 / (1 + distance_km / 1000)  # Normalize by 1000km
            
            feature_row = {
                # Target variable (what we want to predict)
                'clicked': interaction['clicked'],
                'watch_time_minutes': interaction['watch_time_minutes'],
                
                # Location features (key for HOMETOWN)
                'distance_km': distance_km,
                'proximity_boost': proximity_boost,
                'is_local_match': interaction['is_local_match'],
                'is_regional_match': interaction['is_regional_match'],
                'same_city': user_data['city'] == stream_data['city'],
                'same_language': user_data['language'] == stream_data['language'],
                
                # User features
                'user_local_preference': user_data['local_preference_strength'],
                'user_total_watch_hours': user_data['total_watch_hours'],
                'user_timezone_offset': user_data['timezone_offset'],
                
                # Stream features
                'stream_quality_score': stream_data['stream_quality_score'],
                'stream_avg_viewers': stream_data['avg_viewer_count'],
                'creator_followers': stream_data['creator_followers'],
                'is_partnered': stream_data['is_partnered'],
                'mature_content': stream_data['mature_content'],
                
                # Context features
                'page_position': interaction['page_position'],
                'device_type_mobile': interaction['device_type'] == 'mobile',
                'device_type_desktop': interaction['device_type'] == 'desktop',
                
                # Category match
                'category_match': stream_data['category_id'] in user_data['preferred_categories'],
                
                # Time features
                'hour_of_day': interaction['impression_timestamp'].hour,
                'day_of_week': interaction['impression_timestamp'].weekday(),
                
                # IDs for reference
                'user_id': user_id,
                'stream_id': interaction['stream_id'],
                'session_id': interaction['session_id']
            }
            features.append(feature_row)
    
    return pd.DataFrame(features)

@app.command()
def generate_dataset(
    num_users: int = typer.Option(1000, help="Number of users to generate"),
    num_streams: int = typer.Option(500, help="Number of streams to generate"), 
    num_interactions: int = typer.Option(10000, help="Number of interactions to generate"),
    output_dir: str = typer.Option("data", help="Output directory for generated files"),
    train_ratio: float = typer.Option(0.8, help="Ratio of data for training (rest for testing)")
):
    """Generate synthetic dataset for HOMETOWN recommendation scenario."""
    
    typer.echo("🏠 Generating HOMETOWN Scenario Synthetic Dataset")
    typer.echo(f"📊 Users: {num_users:,}, Streams: {num_streams:,}, Interactions: {num_interactions:,}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate core data
    typer.echo("👥 Generating user profiles...")
    users_df = generate_user_profiles(num_users)
    
    typer.echo("📺 Generating stream metadata...")
    streams_df = generate_stream_metadata(num_streams)
    
    typer.echo("🔄 Generating user-stream interactions...")
    interactions_df = generate_interactions(users_df, streams_df, num_interactions)
    
    typer.echo("🎯 Creating HOMETOWN-specific features...")
    features_df = create_hometown_features(interactions_df, users_df, streams_df)
    
    # Train/test split by users to avoid data leakage
    unique_users = features_df['user_id'].unique()
    train_users = set(np.random.choice(unique_users, size=int(len(unique_users) * train_ratio), replace=False))
    
    train_features = features_df[features_df['user_id'].isin(train_users)]
    test_features = features_df[~features_df['user_id'].isin(train_users)]
    
    # Save datasets
    typer.echo("💾 Saving datasets...")
    
    # Core data
    users_df.to_parquet(output_path / "users.parquet", index=False)
    streams_df.to_parquet(output_path / "streams.parquet", index=False) 
    interactions_df.to_parquet(output_path / "interactions.parquet", index=False)
    
    # ML-ready features
    train_features.to_parquet(output_path / "hometown_train.parquet", index=False)
    test_features.to_parquet(output_path / "hometown_test.parquet", index=False)
    
    # Summary statistics
    stats = {
        'dataset_info': {
            'scenario': 'HOMETOWN',
            'generation_timestamp': datetime.now().isoformat(),
            'total_users': len(users_df),
            'total_streams': len(streams_df),
            'total_interactions': len(interactions_df),
            'train_interactions': len(train_features),
            'test_interactions': len(test_features)
        },
        'location_stats': {
            'cities_covered': len(CITIES),
            'avg_distance_km': interactions_df['distance_km'].mean(),
            'local_interaction_rate': (interactions_df['distance_km'] < 100).mean(),
            'regional_interaction_rate': (interactions_df['distance_km'] < 1000).mean()
        },
        'performance_stats': {
            'overall_ctr': interactions_df['clicked'].mean(),
            'local_ctr': interactions_df[interactions_df['distance_km'] < 100]['clicked'].mean(),
            'non_local_ctr': interactions_df[interactions_df['distance_km'] >= 100]['clicked'].mean(),
            'avg_watch_time_local': interactions_df[interactions_df['distance_km'] < 100]['watch_time_minutes'].mean(),
            'avg_watch_time_non_local': interactions_df[interactions_df['distance_km'] >= 100]['watch_time_minutes'].mean()
        },
        'feature_importance_hints': {
            'key_features': [
                'distance_km', 'proximity_boost', 'is_local_match', 
                'user_local_preference', 'same_city', 'same_language',
                'stream_quality_score', 'page_position'
            ],
            'target_variables': ['clicked', 'watch_time_minutes']
        }
    }
    
    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Print summary
    typer.echo("\n✅ Dataset generation complete!")
    typer.echo(f"📁 Files saved to: {output_path}")
    typer.echo(f"📊 Dataset statistics:")
    typer.echo(f"   • Total users: {len(users_df):,}")
    typer.echo(f"   • Total streams: {len(streams_df):,}")
    typer.echo(f"   • Total interactions: {len(interactions_df):,}")
    typer.echo(f"   • Training samples: {len(train_features):,}")
    typer.echo(f"   • Test samples: {len(test_features):,}")
    typer.echo(f"   • Cities covered: {len(CITIES)}")
    typer.echo(f"   • Overall CTR: {interactions_df['clicked'].mean():.3f}")
    typer.echo(f"   • Local CTR: {interactions_df[interactions_df['distance_km'] < 100]['clicked'].mean():.3f}")
    typer.echo(f"   • Non-local CTR: {interactions_df[interactions_df['distance_km'] >= 100]['clicked'].mean():.3f}")
    
    typer.echo("\n🎯 Ready for HOMETOWN scenario training!")
    typer.echo("Use the generated files with your ML pipeline to train location-aware recommendation models.")

if __name__ == "__main__":
    app()
