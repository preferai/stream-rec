"""
Data access layer for loading user and stream data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from geopy.distance import geodesic
from pyqtree import Index
import logging
import numpy as np

from ..models.data_models import User, Stream

logger = logging.getLogger(__name__)


class DataStore:
    """Handles loading and accessing user and stream data, with an efficient spatial index."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._users: Dict[str, User] = {}
        self._streams: Dict[str, Stream] = {}
        self._stream_spatial_index: Optional[Index] = None
        self._load_data()
    
    def _load_data(self):
        """
        Load user and stream data from parquet files.
        Uses df.to_dict('records') for fast loading and builds a spatial index for streams.
        """
        try:
            # --- Optimized User Loading ---
            users_path = self.data_dir / "users.parquet"
            logger.info(f"🔄 Loading users from {users_path}...")
            users_df = pd.read_parquet(users_path)
            user_records = users_df.to_dict('records')
            self._users = {rec['user_id']: User(**rec) for rec in user_records}
            logger.info(f"✅ Loaded {len(self._users):,} users.")

            # --- Optimized Stream Loading ---
            streams_path = self.data_dir / "streams.parquet"
            logger.info(f"🔄 Loading streams from {streams_path}...")
            streams_df = pd.read_parquet(streams_path)
            stream_records = streams_df.to_dict('records')
            self._streams = {rec['stream_id']: Stream(**rec) for rec in stream_records}
            logger.info(f"✅ Loaded {len(self._streams):,} streams.")

            # --- Build Spatial Index for Efficient Geo-Lookups ---
            logger.info("🔨 Building spatial index for streams...")
            # Bbox is the entire world in lat/lon coordinates
            self._stream_spatial_index = Index(bbox=[-180, -90, 180, 90])
            for stream in self._streams.values():
                # Bbox for a point is just (x, y, x, y)
                bbox = (stream.longitude, stream.latitude, stream.longitude, stream.latitude)
                self._stream_spatial_index.insert(item=stream.stream_id, bbox=bbox)
            logger.info("✅ Spatial index built successfully.")

        except FileNotFoundError as e:
            logger.error(f"❌ Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ An error occurred during data loading: {e}")
            raise
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def get_stream(self, stream_id: str) -> Optional[Stream]:
        """Get stream by ID."""
        return self._streams.get(stream_id)
    
    def get_all_streams(self) -> List[Stream]:
        """Get all available streams."""
        return list(self._streams.values())

    def get_streams_in_radius(self, user: User, radius_km: float) -> List[Tuple[Stream, float]]:
        """
        Efficiently get streams within a given radius of a user using the spatial index.
        Returns a list of (stream, distance_km) tuples.
        """
        if not self._stream_spatial_index:
            return []

        # Approximate a bounding box for the search radius to query the index.
        lat_change = radius_km / 111.0  # ~111 km per degree of latitude
        
        # Longitude change depends on latitude
        cos_lat = np.cos(np.radians(user.latitude))
        lon_change = radius_km / (111.0 * cos_lat) if cos_lat > 0 else radius_km / 111.0

        search_bbox = (
            user.longitude - lon_change,
            user.latitude - lat_change,
            user.longitude + lon_change,
            user.latitude + lat_change
        )
        
        # Query the spatial index to get stream IDs within the approximate bounding box
        candidate_stream_ids = self._stream_spatial_index.intersect(search_bbox)
        
        nearby_streams = []
        user_coords = (user.latitude, user.longitude)
        
        # Final filtering: check the precise distance for the candidates
        for stream_id in candidate_stream_ids:
            stream = self.get_stream(stream_id)
            if stream:
                stream_coords = (stream.latitude, stream.longitude)
                distance = geodesic(user_coords, stream_coords).kilometers
                if distance <= radius_km:
                    nearby_streams.append((stream, distance))
                    
        return nearby_streams
    
    def calculate_distance_km(self, user: User, stream: Stream) -> float:
        """Calculate distance between user and stream in kilometers."""
        user_coords = (user.latitude, user.longitude)
        stream_coords = (stream.latitude, stream.longitude)
        return geodesic(user_coords, stream_coords).kilometers
