#!/usr/bin/env python3
"""
Test script to compare basic vs ML-enhanced HOMETOWN recommendations.
"""

import requests
import json


def test_user(user_id: str, max_results: int = 5):
    """Test both endpoints for the same user and display results side by side."""
    
    base_url = "http://localhost:8000"
    
    # Test basic endpoint
    print(f"🏠 Testing user: {user_id}")
    print("=" * 80)
    
    basic_response = requests.post(
        f"{base_url}/v1/scenarios/hometown",
        json={"user_id": user_id, "max_results": max_results}
    )
    
    ml_response = requests.post(
        f"{base_url}/v1/scenarios/hometown-ml", 
        json={"user_id": user_id, "max_results": max_results}
    )
    
    if basic_response.status_code == 200 and ml_response.status_code == 200:
        basic_data = basic_response.json()
        ml_data = ml_response.json()
        
        print("📊 COMPARISON RESULTS:")
        print(f"{'Rank':<4} {'Basic Algorithm':<50} {'ML-Enhanced Algorithm':<50}")
        print("-" * 110)
        
        for i in range(min(len(basic_data['streams']), len(ml_data['streams']))):
            basic_stream = basic_data['streams'][i]
            ml_stream = ml_data['streams'][i]
            
            basic_info = f"{basic_stream['stream_id']} (score: {basic_stream['score']:.3f})"
            ml_info = f"{ml_stream['stream_id']} (score: {ml_stream['score']:.3f})"
            
            print(f"{i+1:<4} {basic_info:<50} {ml_info:<50}")
        
        # Calculate score differences
        basic_scores = [s['score'] for s in basic_data['streams']]
        ml_scores = [s['score'] for s in ml_data['streams']]
        
        avg_basic = sum(basic_scores) / len(basic_scores)
        avg_ml = sum(ml_scores) / len(ml_scores)
        
        print(f"\n📈 AVERAGE SCORES:")
        print(f"Basic Algorithm: {avg_basic:.3f}")
        print(f"ML-Enhanced:     {avg_ml:.3f}")
        print(f"ML Boost:        +{avg_ml - avg_basic:.3f}")
        
    else:
        print(f"❌ Error - Basic: {basic_response.status_code}, ML: {ml_response.status_code}")
        if basic_response.status_code != 200:
            print(f"Basic error: {basic_response.text}")
        if ml_response.status_code != 200:
            print(f"ML error: {ml_response.text}")


def main():
    """Test multiple users."""
    print("🎯 HOMETOWN Algorithm Comparison Tool")
    print("=====================================\n")
    
    # Test different users
    test_users = ["user_000001", "user_000010", "user_000020"]
    
    for user_id in test_users:
        test_user(user_id, max_results=5)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
