"""
FastAPI application for streaming recommendation scenarios.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from ..models.data_models import HometownRequest, HometownResponse
from ..services.data_store import DataStore
from ..services.hometown_recommender import HometownRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stream Recommendation API",
    description="API for streaming platform recommendation scenarios",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data store and recommender
try:
    data_store = DataStore()
    hometown_recommender = HometownRecommender(data_store)
    logger.info("✅ Services initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize services: {e}")
    data_store = None
    hometown_recommender = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Stream Recommendation API", 
        "status": "running",
        "available_scenarios": ["hometown"]
    }


@app.post("/v1/scenarios/hometown", response_model=HometownResponse)
async def hometown_scenario(request: HometownRequest):
    """
    HOMETOWN Scenario: Location-based stream recommendations.
    
    Orders the front-page grid so that live channels from the user's own 
    city/region appear first using proximity boost algorithm.
    
    Algorithm:
    - Fetch user's city/geo coordinates
    - Calculate proximity boost B = 1 / (1 + distance_km) for each stream  
    - Combine with base quality scores
    - Sort by final score so local channels surface first
    
    Args:
        request: Contains user_id
        
    Returns:
        List of recommended streams with location information
    """
    if not hometown_recommender:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service not available"
        )
    
    try:
        logger.info(f"🏠 Processing HOMETOWN request for user: {request.user_id}")
        
        # Generate recommendations using HOMETOWN algorithm
        recommendations = hometown_recommender.recommend_streams(
            user_id=request.user_id,
            max_results=20
        )
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        
        return HometownResponse(streams=recommendations)
        
    except ValueError as e:
        logger.warning(f"⚠️ User not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/v1/scenarios/hometown/stats")
async def hometown_stats():
    """Get statistics about the HOMETOWN recommendation system."""
    if not data_store:
        raise HTTPException(status_code=503, detail="Data store not available")
    
    try:
        all_streams = data_store.get_all_streams()
        
        # Calculate city distribution
        cities = {}
        for stream in all_streams:
            cities[stream.city] = cities.get(stream.city, 0) + 1
        
        stats = {
            "total_streams": len(all_streams),
            "total_users": len(data_store._users),
            "cities_covered": len(cities),
            "top_cities": dict(sorted(cities.items(), key=lambda x: x[1], reverse=True)[:10]),
            "algorithm": "Proximity boost B = 1 / (1 + distance_km)"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
