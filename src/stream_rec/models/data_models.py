"""
Data models for the HOMETOWN recommendation scenario.
"""

from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class User(BaseModel):
    """User profile with location information."""
    user_id: str
    city: str
    latitude: float
    longitude: float
    language: str
    timezone_offset: int
    total_watch_hours: float
    local_preference_strength: float
    preferred_categories: List[str]


class Stream(BaseModel):
    """Stream metadata with creator location."""
    stream_id: str
    creator_id: str
    city: str
    latitude: float
    longitude: float
    language: str
    category_id: str
    title: str
    description: str
    avg_viewer_count: int
    creator_followers: int
    stream_quality_score: float
    is_partnered: bool
    mature_content: bool
    tags: List[str]


class HometownRequest(BaseModel):
    """Request schema for HOMETOWN scenario endpoint."""
    user_id: str


class StreamRecommendation(BaseModel):
    """Individual stream recommendation with location info."""
    stream_id: str
    city: str
    score: Optional[float] = None
    distance_km: Optional[float] = None
    proximity_boost: Optional[float] = None


class HometownResponse(BaseModel):
    """Response schema for HOMETOWN scenario endpoint."""
    streams: List[StreamRecommendation]
