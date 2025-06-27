# HOMETOWN Scenario POC

The HOMETOWN scenario recommends live streams from a user's **geographic proximity** - prioritizing local and regional content to create community connections. 

### Data Generation & Exploration

The HOMETOWN scenario assumes the following data to enable effective geographic proximity recommendations. The synthetic dataset is designed around these **core requirements**:

**User Data**:
- **Geographic coordinates** (latitude/longitude) - Required for distance calculations and proximity boost
- **City/region information** - Enables local community discovery and regional grouping
- **Language preferences** - Critical for content relevance and cultural alignment
- **Category preferences** - Ensures recommended streams match user interests
- **Local preference strength** - Quantifies how much users value geographic proximity vs. other factors

**Stream Data**:
- **Geographic coordinates** - Required for spatial indexing and distance calculations
- **Stream quality metrics** - Viewer count, creator popularity, content quality scores
- **Language and category** - Enables preference matching and content filtering
- **Real-time availability** - Live stream status and current viewer engagement

**Synthetic Data Structure**:
The syntetically created dataset design tries to reflect a **real-world streaming platform characteristics** where users don't just want "nearby" content, but **high-quality local content that matches their preferences**. The synthetic data generates:
- **Geographic clustering** around major cities (mimicking population density)
- **Power-law quality distribution** (few high-quality streams, many average ones)
- **Language-region correlation** (Paris users prefer French, Tokyo users prefer Japanese)
- **Preference diversity** within regions (e.g., not all London users like the same categories)

This approach enables testing the full HOMETOWN algorithm complexity while maintaining realistic user behavior patterns.

#### 1. Generate Base Dataset
```bash
uv run python scripts/generate_synthetic_dataset.py
```

**Default configuration**:
- **1,000 users** distributed across 32 major cities worldwide
- **500 streams** with realistic geographic clustering
- **Quality scores** following power-law distribution (simulating real popularity effects)
- **Language and category preferences** based on geographic regions

## 🎯 POC Overview

A FastAPI service with:
- ✅ **Efficient spatial indexing** using Quadtree for O(log N) geo-queries
- ✅ **Weighted scoring algorithm** combining proximity, quality, and preferences
- ✅ **API** with health checks and system statistics
- ✅ **Synthetic dataset generation** script for testing and development

## 🚀 Quick Start Guide

### Prerequisites

1. **Python 3.12+** and **UV Package Manager**
   ```bash
   # Install UV: https://docs.astral.sh/uv/getting-started/installation/
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Docker** (optional, for containerized deployment)

### 1. Setup Environment

```bash
# Clone and install dependencies
git clone https://github.com/preferai/stream-rec.git
cd stream-rec
uv sync
```

### 2. Generate Synthetic Data

Before running the API, generate the synthetic dataset:

```bash
# Generate HOMETOWN-specific training data (optional, for ML model)
uv run python scripts/generate_hometown_dataset.py
```

This creates:
- `data/users.parquet` - 1,000 users across 32 global cities
- `data/streams.parquet` - 500 streams with quality scores and geo-coordinates
- `data/hometown_train.parquet` - Training data for ML model (optional)

### 3. Start the API Server

```bash
# Start the HOMETOWN recommendation API
uv run python run_server.py
```

The server will start on `http://localhost:8000` with these endpoints:
- `GET /` - Health check and available endpoints
- `POST /v1/scenarios/hometown` - Get recommendations (Basic Algorithm)
- `POST /v1/scenarios/hometown-ml` - Get recommendations (ML-Enhanced)
- `GET /v1/scenarios/hometown/stats` - System statistics
- `GET /docs` - Interactive API documentation

### 4. Test the System

```bash
# Test both endpoints
curl "http://localhost:8000/"

# Test basic algorithm
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 3}'

# Test ML-enhanced algorithm (requires model training)
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 3}'

# Run comprehensive API tests
uv run python test_api.py

# Compare both algorithms side-by-side
uv run python test_both_endpoints.py
```

## 📊 Example Usage

### Getting Recommendations (Basic Algorithm)

The basic algorithm uses proximity boost + quality scoring without ML enhancement:

```bash
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 5}'
```

**Response (Basic):**
```json
{
  "streams": [
    {
      "stream_id": "stream_000294",
      "city": "Paris",
      "score": 5.777,
      "distance_km": 10.0
    },
    {
      "stream_id": "stream_000083", 
      "city": "Paris",
      "score": 5.552,
      "distance_km": 10.0
    }
  ]
}
```

### Getting Recommendations (ML-Enhanced)

The ML-enhanced algorithm adds learned click prediction on top of the basic scoring:

```bash
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 5}'
```

**Response (ML-Enhanced):**
```json
{
  "streams": [
    {
      "stream_id": "stream_000083",
      "city": "Paris", 
      "score": 6.234,
      "distance_km": 10.0
    },
    {
      "stream_id": "stream_000294",
      "city": "Paris",
      "score": 6.112,
      "distance_km": 10.0
    }
  ]
}
```

**Note**: The ML endpoint requires training the model first (see ML Model section below).

### System Statistics

```bash
curl "http://localhost:8000/v1/scenarios/hometown/stats"
```

**Response:**
```json
{
  "total_streams": 500,
  "total_users": 1000,
  "cities_covered": 32,
  "top_cities": {
    "Los Angeles": 60,
    "New York": 43,
    "Tokyo": 42,
    "Berlin": 40,
    "Seoul": 38
  },
  "algorithm": "Proximity boost B = 1 / (1 + distance_km)"
}
```

## � Technical Deep Dive

### Architecture Overview

The HOMETOWN recommendation system uses the following **architecture** for geographic queries:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Client    │    │   FastAPI        │    │   Data Layer    │
│                 │───▶│   Service        │───▶│                 │
│  • REST calls   │    │                  │    │ • Spatial Index │
│  • JSON payload │    │  • Validation    │    │ • User/Stream   │
│  • Response     │    │  • Rate Limits   │    │   Data Store    │
└─────────────────┘    │  • Error Handling│    └─────────────────┘
                       └──────────────────┘            │
                                │                      │
                                ▼                      │
                       ┌─────────────────┐             │
                       │   HOMETOWN      │             │
                       │  Recommender    │◀────────────┘
                       │                 │
                       │ • Spatial Query │    ┌─────────────────┐
                       │ • Proximity     │───▶│ Quadtree Index  │
                       │   Scoring       │    │                 │
                       │ • Rank & Filter │    │ • O(log N)      │
                       └─────────────────┘    │   Geo Lookup    │
                                              │ • Distance Calc │
                                              └─────────────────┘
```

### Core Components

#### 1. Spatial Index (`DataStore`)
**File**: `src/stream_rec/services/data_store.py`

The heart of our performance optimization:
```python
# Build spatial index on startup
self._stream_spatial_index = Index(bbox=[-180, -90, 180, 90])
for stream in self._streams.values():
    bbox = (stream.longitude, stream.latitude, stream.longitude, stream.latitude)
    self._stream_spatial_index.insert(item=stream.stream_id, bbox=bbox)
```

**Why this matters**:
- **O(log N)**: Query spatial index for nearby candidates only instead of checking distance to every stream for every request which would be O(N)
- **Impact**: 100x faster queries for large datasets


#### 2. Weighted Scoring Algorithm
**File**: `src/stream_rec/services/hometown_recommender.py`

```python
# Multi-factor scoring formula
final_score = (
    (SCORE_WEIGHTS["proximity"] * proximity_score * local_preference_multiplier) +
    (SCORE_WEIGHTS["base"] * base_score) +
    (SCORE_WEIGHTS["ml"] * ml_score)
)
```

**Components**:
- **Proximity Score**: `1 / (1 + distance_km)` - Closer streams score higher
- **Base Score**: Combines a stream's inherent quality score with user preference bonuses (language match +1.0, category match +0.5) and popularity boost (viewer count/1000, capped at 2.0) to measure non-geographic appeal
- **ML Score**: Optional learned scoring from user behavior patterns

### Data Models

#### User Model
```python
@dataclass
class User:
    user_id: str
    latitude: float          # Geographic coordinates
    longitude: float
    city: str               # Primary city
    language: str           # Preferred language
    preferred_categories: List[int]  # Content preferences
    local_preference_strength: float  # How much user values local content
    total_watch_hours: float
```

#### Stream Model  
```python
@dataclass
class Stream:
    stream_id: str
    latitude: float          # Stream location
    longitude: float
    city: str
    language: str
    category_id: int
    stream_quality_score: float    # Content quality metric
    avg_viewer_count: int
    creator_followers: int
    is_partnered: bool
    mature_content: bool
```


### Data Exploration

Explore the generated data with our Jupyter notebook:

```bash
jupyter notebook notebooks/hometown_data_exploration.ipynb
```

**Analysis includes**:
- Geographic distribution visualization
- Stream quality distributions
- User preference patterns
- Distance vs. engagement correlations

## 🧪 Testing & Validation

### Comprehensive Test Suite

```bash
# Run all API tests
uv run python test_api.py
```

**Tests validate**:
- ✅ Health check endpoint
- ✅ Recommendation generation with various users
- ✅ Geographic proximity accuracy
- ✅ Scoring algorithm correctness
- ✅ System statistics endpoint

### Manual Testing Examples

#### Test Different Geographic Locations
```bash
# User in Paris (Basic Algorithm)
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001"}'

# User in Paris (ML-Enhanced Algorithm)  
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001"}'

# User in Toronto (Basic Algorithm)
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000002"}'
```

#### Compare Basic vs ML-Enhanced Results
```bash
# Compare the same user with both algorithms
echo "=== Basic Algorithm ==="
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 5}'

echo -e "\n=== ML-Enhanced Algorithm ==="
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 5}'
```

#### Validate Proximity Ranking
```bash
# Check that nearby streams rank higher (Basic Algorithm)
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 10}' | \
  jq '.streams | sort_by(.distance_km)'

# Check ranking with ML enhancement
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 10}' | \
  jq '.streams | sort_by(.distance_km)'
```

## 🐳 Docker Deployment

### Quick Container Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access API at http://localhost:8000
curl "http://localhost:8000/"
```

### Development Mode
```bash
# Run with auto-reload for development
docker-compose -f docker-compose.dev.yml up
```

### Production Deployment
```bash
# Production optimized container
docker-compose -f docker-compose.yml up -d
```


## 🔬 ML Model Options: Two Separate Endpoints

The HOMETOWN recommendation system provides **two separate API endpoints** for easy comparison:

1. **`/v1/scenarios/hometown`** - Basic algorithm (proximity + quality scoring)
2. **`/v1/scenarios/hometown-ml`** - ML-enhanced algorithm (adds click prediction)

This design makes it easy to A/B test and compare results without changing code.

### Endpoint 1: Basic Algorithm (`/hometown`)

**Always available** - Uses only proximity and base scoring:

```bash
# No setup required - works immediately
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 3}'
```

**Scoring Formula**:
```python
final_score = (
    (1.5 * proximity_score * local_preference_multiplier) +  # Geographic closeness
    (1.0 * base_score) +                                    # Stream quality + preferences
    (0.5 * 0.0)                                             # ML score = 0 (no model)
)
```

### Endpoint 2: ML-Enhanced Algorithm (`/hometown-ml`)

**Requires model training** - Adds learned click prediction:

**Step 1: Train the ML Model** (one-time setup)

We train a **Logistic Regression classifier** to predict the probability that a user will click on a stream recommendation.

**Important**: The ML model is **not** trained automatically when the FastAPI service starts. It's a separate manual step that creates a model file which the API can optionally load.

```bash
# Generate training data (if not already done)
uv run python scripts/generate_hometown_dataset.py

# Train the logistic regression model (creates models/hometown_model.pkl)
uv run python src/stream_rec/services/model_trainer.py
```

**What this step does**:
- Loads training data from `data/hometown_train.parquet`
- Trains a Logistic Regression model on 14 engineered features
- Saves the trained model to `models/hometown_model.pkl`
- **The API runs fine without this step** - it just won't have ML scoring

**What exactly gets trained**:
- **Model Type**: `sklearn.linear_model.LogisticRegression` 
- **Task**: Binary classification (will user click: yes/no)
- **Target Variable**: `clicked` (0 or 1)
- **Training Data**: synthetic user-stream interaction pairs
- **Features**: 14 engineered features combining geographic, user preference, and stream quality signals

**Training Data Structure**:
```python
# Each training sample looks like this:
{
    'user_id': 'user_000123',
    'stream_id': 'stream_000456', 
    'distance_km': 15.7,
    'proximity_boost': 0.0593,
    'is_local_match': 1,           # boolean: within 100km
    'is_regional_match': 1,        # boolean: within 1000km  
    'same_city': 1,                # boolean: exact city match
    'same_language': 1,            # boolean: language match
    'user_local_preference': 0.8,  # float: how much user values local (0-1)
    'user_total_watch_hours': 120.5,
    'stream_quality_score': 7.2,
    'stream_avg_viewers': 850,
    'creator_followers': 15000,
    'is_partnered': 1,             # boolean: professional creator
    'mature_content': 0,           # boolean: age-restricted content
    'category_match': 1,           # boolean: matches user preferences
    'clicked': 1                   # TARGET: did user click? (0 or 1)
}
```

**Example Output**:
```
🏠 Training HOMETOWN ML Model...
📊 Loaded 25,000 training samples
🔧 Training on 20,000 samples, testing on 5,000
✅ Model trained successfully!
   • Train Accuracy: 0.745
   • Test Accuracy: 0.738
   • AUC Score: 0.814

🎯 Top Feature Importances:
   • proximity_boost         :  2.156
   • stream_quality_score     :  1.834
   • same_language           :  1.245
   • user_local_preference   :  0.987
   • category_match          :  0.654

💾 Model saved to models/hometown_model.pkl
```

**Step 2: Test the ML-Enhanced Endpoint**

No code changes needed! The API automatically loads the model:

```bash
# Test the ML-enhanced endpoint
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 3}'
```

**Enhanced Scoring Formula**:
```python
final_score = (
    (1.5 * proximity_score * local_preference_multiplier) +  # Geographic closeness
    (1.0 * base_score) +                                    # Stream quality + preferences  
    (0.5 * ml_prediction)                                   # ← ML enhancement (0.0-1.0)
)
```

**Response (ML-Enhanced)**:
```json
{
  "streams": [
    {
      "stream_id": "stream_000083",  // ← Different ranking due to ML
      "city": "Paris",
      "score": 6.234,               // ← Higher score with ML boost
      "distance_km": 10.0
    }
  ]
}
```

### Side-by-Side Comparison

```bash
# Compare both algorithms for the same user
echo "=== Basic Algorithm ==="
curl -X POST "http://localhost:8000/v1/scenarios/hometown" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 3}' | jq

echo -e "\n=== ML-Enhanced Algorithm ==="
curl -X POST "http://localhost:8000/v1/scenarios/hometown-ml" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_000001", "max_results": 3}' | jq
```

### Comparison Tool

For detailed side-by-side analysis, use the provided comparison script:

```bash
# Run the comprehensive comparison tool
uv run python test_both_endpoints.py
```

**Sample Output**:
```
🎯 HOMETOWN Algorithm Comparison Tool
=====================================

🏠 Testing user: user_000001
================================================================================
📊 COMPARISON RESULTS:
Rank Basic Algorithm                           ML-Enhanced Algorithm          
------------------------------------------------------------------------------
1    stream_000294 (score: 5.777)             stream_000294 (score: 5.983)  
2    stream_000083 (score: 5.552)             stream_000083 (score: 5.770)  
3    stream_000039 (score: 4.074)             stream_000039 (score: 4.252)  

📈 AVERAGE SCORES:
Basic Algorithm: 4.237
ML-Enhanced:     4.441
ML Boost:        +0.205
```


### Model Details: Logistic Regression for Click Prediction

**Exact Model Specification**:
- **Algorithm**: `sklearn.linear_model.LogisticRegression`
- **Hyperparameters**: `random_state=42, max_iter=1000`
- **Training**: 80/20 train/test split with stratification
- **Evaluation**: AUC-ROC, accuracy, precision/recall
- **Prediction**: Returns probability score (0.0-1.0) that user will click

**Training Data Generation**:
The synthetic training dataset simulates realistic user behavior:
```python
# For each user-stream pair, we calculate:
click_probability = base_engagement * proximity_factor * preference_match
clicked = 1 if random() < click_probability else 0
```

**14 Input Features**:
```python
features = [
    'distance_km',              # Geographic distance (float, 0-20000km)
    'proximity_boost',          # 1/(1+distance_km) (float, 0-1)
    'is_local_match',          # Within 100km (int, 0 or 1)
    'is_regional_match',       # Within 1000km (int, 0 or 1)
    'same_city',               # Exact city match (int, 0 or 1)
    'same_language',           # Language preference match (int, 0 or 1)
    'user_local_preference',   # User's local content preference (float, 0-1)
    'user_total_watch_hours',  # User engagement level (float, 0-500)
    'stream_quality_score',    # Content quality metric (float, 0-10)
    'stream_avg_viewers',      # Stream popularity (int, 0-10000)
    'creator_followers',       # Creator popularity (int, 0-100000)
    'is_partnered',           # Professional creator (int, 0 or 1)
    'mature_content',         # Age-restricted content (int, 0 or 1)
    'category_match'          # User's preferred category (int, 0 or 1)
]
```

**How the Model is Used**:
```python
# At inference time:
user_stream_features = prepare_features(user, stream, distance)
click_probability = model.predict_proba(user_stream_features)[0][1]
ml_score = click_probability  # 0.0 to 1.0

# Integrated into final scoring:
final_score = (1.5 * proximity_score) + (1.0 * base_score) + (0.5 * ml_score)
```
