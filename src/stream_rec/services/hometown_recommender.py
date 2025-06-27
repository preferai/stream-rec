"""
Core recommendation engine for the HOMETOWN scenario.
"""

import numpy as np
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from geopy.distance import geodesic

from ..models.data_models import User, Stream, StreamRecommendation
from .data_store import DataStore

# --- Constants ---
# Defines the search radius for the initial candidate set from the spatial index.
# This value represents a balance between capturing all relevant "hometown"
# streams and keeping the candidate set small for performance. 250km covers
# a large metropolitan area and its surroundings.
SEARCH_RADIUS_KM = 250.0

# Weights for the final scoring formula. These can be tuned to adjust the
# importance of each component in the recommendation.
# w1: Proximity Score - How much to value physical closeness.
# w2: Base Score - How much to value stream quality, language match, etc.
# w3: ML Score - How much to value the prediction from the ML model, if available.
SCORE_WEIGHTS = {
    "proximity": 1.5,
    "base": 1.0,
    "ml": 0.5
}


class HometownRecommender:
    """
    HOMETOWN scenario recommender that uses a spatial index for efficient,
    scalable, and production-ready location-based ranking.

    Algorithm:
    1.  Efficiently fetch candidate streams within a defined radius (e.g., 250km)
        of the user using a Quadtree spatial index.
    2.  For each candidate, calculate a weighted final score based on:
        a. Proximity Boost: A score inversely proportional to distance.
        b. Base Score: A measure of stream quality and user preference match.
        c. ML Score: A predictive score from an optional trained model.
    3.  Sort streams by the final weighted score.
    """

    def __init__(self, data_store: DataStore, model_path: str = None):
        self.data_store = data_store
        self.model = None
        
        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)

    def calculate_proximity_boost(self, distance_km: float) -> float:
        """
        Calculate proximity boost using the HOMETOWN formula.
        """
        if distance_km < 0:
            return 0.0
        return 1.0 / (1.0 + distance_km)

    def calculate_base_score(self, user: User, stream: Stream) -> float:
        """
        Calculate base recommendation score before location boost.
        """
        score = stream.stream_quality_score
        if user.language == stream.language:
            score += 0.5
        if stream.category_id in user.preferred_categories:
            score += 1.0
        popularity_bonus = min(stream.creator_followers / 10000, 1.0)
        score += popularity_bonus
        if stream.is_partnered:
            score += 0.3
        return score

    def calculate_ml_score(self, user: User, stream: Stream, distance_km: float) -> float:
        """
        Calculate ML-based score if a model is available.
        """
        if self.model is None:
            return 0.0
        
        features = np.array([[
            distance_km,
            self.calculate_proximity_boost(distance_km),
            1 if distance_km < 100 else 0,
            1 if distance_km < 1000 else 0,
            1 if user.city == stream.city else 0,
            1 if user.language == stream.language else 0,
            user.local_preference_strength,
            user.total_watch_hours,
            stream.stream_quality_score,
            stream.avg_viewer_count,
            stream.creator_followers,
            1 if stream.is_partnered else 0,
            1 if stream.mature_content else 0,
            1 if stream.category_id in user.preferred_categories else 0
        ]])
        
        try:
            return self.model.predict_proba(features)[0][1]
        except Exception:
            return 0.0

    def recommend_streams(self, user_id: str, max_results: int = 20) -> List[StreamRecommendation]:
        """
        Generate HOMETOWN recommendations using the efficient spatial index.

        Process:
        1. Get user and their location.
        2. Query the DataStore's spatial index for streams within SEARCH_RADIUS_KM.
        3. Score this smaller candidate set using a weighted formula.
        4. Sort and return the top recommendations.
        """
        user = self.data_store.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # 1. Efficiently get candidate streams using the spatial index
        candidate_streams_with_dist = self.data_store.get_streams_in_radius(user, SEARCH_RADIUS_KM)

        # 2. Calculate scores for the candidate set
        scored_streams = []
        for stream, distance_km in candidate_streams_with_dist:
            
            # Calculate the three main components of the score
            proximity_score = self.calculate_proximity_boost(distance_km)
            base_score = self.calculate_base_score(user, stream)
            ml_score = self.calculate_ml_score(user, stream, distance_km)

            # Apply user's explicit local preference strength
            local_preference_multiplier = 1.0 + user.local_preference_strength
            
            # Combine scores using the defined weights
            final_score = (
                (SCORE_WEIGHTS["proximity"] * proximity_score * local_preference_multiplier) +
                (SCORE_WEIGHTS["base"] * base_score) +
                (SCORE_WEIGHTS["ml"] * ml_score)
            )

            scored_streams.append({
                "stream": stream,
                "distance_km": distance_km,
                "final_score": final_score
            })

        # 3. Sort by final score (highest first)
        scored_streams.sort(key=lambda x: x["final_score"], reverse=True)

        # 4. Convert to response format
        recommendations = []
        for item in scored_streams[:max_results]:
            stream = item["stream"]
            recommendation = StreamRecommendation(
                stream_id=stream.stream_id,
                city=stream.city,
                score=round(item["final_score"], 3),
                distance_km=round(item["distance_km"], 1)
            )
            recommendations.append(recommendation)

        return recommendations
