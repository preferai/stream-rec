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

# Initialize data store and recommenders
try:
    data_store = DataStore()
    # Basic recommender (no ML model)
    hometown_recommender_basic = HometownRecommender(data_store)
    # ML-enhanced recommender (loads model if available)
    hometown_recommender_ml = HometownRecommender(data_store, model_path="models/hometown_model.pkl")
    logger.info("✅ Services initialized successfully")
    logger.info(f"🤖 ML model loaded: {hometown_recommender_ml.model is not None}")
except Exception as e:
    logger.error(f"❌ Failed to initialize services: {e}")
    data_store = None
    hometown_recommender_basic = None
    hometown_recommender_ml = None


@app.get("/")
async def root():
    """Health check endpoint."""
    ml_available = hometown_recommender_ml and hometown_recommender_ml.model is not None
    return {
        "message": "Stream Recommendation API", 
        "status": "running",
        "available_scenarios": {
            "hometown": "/v1/scenarios/hometown (Basic proximity + quality scoring)",
            "hometown-ml": f"/v1/scenarios/hometown-ml (ML-enhanced) - {'Available' if ml_available else 'Model not loaded'}"
        }
    }


@app.post("/v1/scenarios/hometown", response_model=HometownResponse)
async def hometown_scenario(request: HometownRequest):
    """
    HOMETOWN Scenario: Location-based stream recommendations (Basic Algorithm).
    
    Uses proximity boost and base scoring without ML enhancement.
    Orders the front-page grid so that live channels from the user's own 
    city/region appear first using proximity boost algorithm.
    
    Algorithm:
    - Fetch user's city/geo coordinates
    - Calculate proximity boost B = 1 / (1 + distance_km) for each stream  
    - Combine with base quality scores (no ML)
    - Sort by final score so local channels surface first
    
    Args:
        request: Contains user_id and max_results
        
    Returns:
        List of recommended streams with location information
    """
    if not hometown_recommender_basic:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service not available"
        )
    
    try:
        logger.info(f"🏠 Processing HOMETOWN (Basic) request for user: {request.user_id}")
        
        # Generate recommendations using basic HOMETOWN algorithm
        recommendations = hometown_recommender_basic.recommend_streams(
            user_id=request.user_id,
            max_results=request.max_results
        )
        
        logger.info(f"✅ Generated {len(recommendations)} basic recommendations")
        
        return HometownResponse(streams=recommendations)
        
    except ValueError as e:
        logger.warning(f"⚠️ User not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/scenarios/hometown-ml", response_model=HometownResponse)
async def hometown_scenario_ml(request: HometownRequest):
    """
    HOMETOWN Scenario: Location-based stream recommendations (ML-Enhanced).
    
    Uses proximity boost, base scoring, and ML enhancement for click prediction.
    Orders the front-page grid so that live channels from the user's own 
    city/region appear first using proximity boost + ML scoring.
    
    Algorithm:
    - Fetch user's city/geo coordinates
    - Calculate proximity boost B = 1 / (1 + distance_km) for each stream  
    - Combine with base quality scores + ML prediction scores
    - Sort by final weighted score so local channels surface first
    
    Args:
        request: Contains user_id and max_results
        
    Returns:
        List of recommended streams with location information and ML scoring
    """
    if not hometown_recommender_ml:
        raise HTTPException(
            status_code=503, 
            detail="ML recommendation service not available"
        )
    
    try:
        logger.info(f"🤖 Processing HOMETOWN (ML) request for user: {request.user_id}")
        
        # Check if ML model is loaded
        if hometown_recommender_ml.model is None:
            raise HTTPException(
                status_code=503, 
                detail="ML model not loaded. Train model first using: uv run python src/stream_rec/services/model_trainer.py"
            )
        
        # Generate recommendations using ML-enhanced HOMETOWN algorithm
        recommendations = hometown_recommender_ml.recommend_streams(
            user_id=request.user_id,
            max_results=request.max_results
        )
        
        logger.info(f"✅ Generated {len(recommendations)} ML-enhanced recommendations")
        
        return HometownResponse(streams=recommendations)
        
    except ValueError as e:
        logger.warning(f"⚠️ User not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error generating ML recommendations: {e}")
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
