#!/usr/bin/env python3
"""
Test script for the HOMETOWN recommendation API.
"""

import requests
import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_hometown_recommendation(user_id: str = "user_000001"):
    """Test the HOMETOWN recommendation endpoint."""
    print(f"🏠 Testing HOMETOWN recommendation for user: {user_id}")
    
    payload = {"user_id": user_id}
    response = requests.post(f"{BASE_URL}/v1/scenarios/hometown", json=payload)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        streams = data["streams"]
        print(f"✅ Received {len(streams)} recommendations")
        
        # Show top 5 recommendations
        print("\n🎯 Top 5 Recommendations:")
        for i, stream in enumerate(streams[:5], 1):
            print(f"{i}. Stream {stream['stream_id']}")
            print(f"   📍 City: {stream['city']}")
            print(f"   📊 Score: {stream.get('score', 'N/A')}")
            print(f"   📏 Distance: {stream.get('distance_km', 'N/A')} km")
            print(f"   🚀 Proximity Boost: {stream.get('proximity_boost', 'N/A')}")
            print()
    else:
        print(f"❌ Error: {response.text}")
    
    return response.status_code == 200


def test_stats():
    """Test the stats endpoint."""
    print("📊 Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/v1/scenarios/hometown/stats")
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print("✅ System Statistics:")
        print(f"   • Total Users: {stats.get('total_users', 'N/A'):,}")
        print(f"   • Total Streams: {stats.get('total_streams', 'N/A'):,}")
        print(f"   • Cities Covered: {stats.get('cities_covered', 'N/A')}")
        print(f"   • Algorithm: {stats.get('algorithm', 'N/A')}")
        
        top_cities = stats.get('top_cities', {})
        print(f"\n🏙️ Top Cities:")
        for city, count in list(top_cities.items())[:5]:
            print(f"   • {city}: {count} streams")
    else:
        print(f"❌ Error: {response.text}")
    
    return response.status_code == 200


def main():
    """Run all tests."""
    print("🧪 Testing HOMETOWN Recommendation API")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("HOMETOWN Recommendation", test_hometown_recommendation),
        ("System Stats", test_stats)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, "❌ ERROR"))
        print("-" * 30)
    
    print("\n📋 Test Results:")
    for test_name, result in results:
        print(f"   {test_name:<25}: {result}")
    
    print(f"\n🎯 API Documentation: {BASE_URL}/docs")


if __name__ == "__main__":
    main()
