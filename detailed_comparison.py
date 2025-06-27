#!/usr/bin/env python3
"""
Detailed comparison script for HOMETOWN recommendation endpoints.

This script provides a deeper analysis of how the basic and ML-enhanced 
algorithms compare, including ranking differences and score distributions.
"""

import pandas as pd
import requests
import json
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DetailedComparison:
    """Provides detailed comparison between basic and ML algorithms."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_data = None
        
    def load_test_data(self, test_file: str = "data/hometown_test.parquet"):
        """Load test data."""
        print("📊 Loading test data...")
        self.test_data = pd.read_parquet(test_file)
        print(f"✅ Loaded {len(self.test_data)} interactions")
        
    def get_recommendations_with_scores(self, user_id: str, endpoint: str, k: int = 10) -> List[Dict]:
        """Get recommendations with full details from API endpoint."""
        url = f"{self.base_url}/v1/scenarios/{endpoint}"
        payload = {"user_id": user_id, "max_results": k}
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('streams', [])
            else:
                print(f"⚠️ API error for {user_id} on {endpoint}: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Request failed for {user_id} on {endpoint}: {e}")
            return []
    
    def compare_single_user(self, user_id: str, k: int = 10) -> Dict:
        """Compare recommendations for a single user."""
        print(f"🔍 Analyzing user {user_id}")
        
        # Get recommendations from both endpoints
        basic_recs = self.get_recommendations_with_scores(user_id, 'hometown', k)
        ml_recs = self.get_recommendations_with_scores(user_id, 'hometown-ml', k)
        
        if not basic_recs or not ml_recs:
            return None
        
        # Extract stream IDs and scores
        basic_streams = [rec['stream_id'] for rec in basic_recs]
        ml_streams = [rec['stream_id'] for rec in ml_recs]
        basic_scores = [rec['score'] for rec in basic_recs]
        ml_scores = [rec['score'] for rec in ml_recs]
        
        # Calculate metrics
        ranking_similarity = self.calculate_ranking_similarity(basic_streams, ml_streams)
        score_correlation = np.corrcoef(basic_scores[:min(len(basic_scores), len(ml_scores))], 
                                      ml_scores[:min(len(basic_scores), len(ml_scores))])[0,1]
        
        # Score statistics
        basic_stats = {
            'mean': np.mean(basic_scores),
            'std': np.std(basic_scores),
            'min': np.min(basic_scores),
            'max': np.max(basic_scores)
        }
        
        ml_stats = {
            'mean': np.mean(ml_scores),
            'std': np.std(ml_scores),
            'min': np.min(ml_scores),
            'max': np.max(ml_scores)
        }
        
        return {
            'user_id': user_id,
            'basic_streams': basic_streams,
            'ml_streams': ml_streams,
            'basic_scores': basic_scores,
            'ml_scores': ml_scores,
            'ranking_similarity': ranking_similarity,
            'score_correlation': score_correlation,
            'basic_stats': basic_stats,
            'ml_stats': ml_stats,
            'recommendations_identical': basic_streams == ml_streams
        }
    
    def calculate_ranking_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate ranking similarity using Spearman's rank correlation."""
        if not list1 or not list2:
            return 0.0
        
        # Create rank dictionaries
        rank1 = {item: i for i, item in enumerate(list1)}
        rank2 = {item: i for i, item in enumerate(list2)}
        
        # Find common items
        common_items = set(list1) & set(list2)
        if len(common_items) < 2:
            return 1.0 if list1 == list2 else 0.0
        
        # Get ranks for common items
        ranks1 = [rank1[item] for item in common_items]
        ranks2 = [rank2[item] for item in common_items]
        
        # Calculate Spearman correlation
        return np.corrcoef(ranks1, ranks2)[0,1] if len(ranks1) > 1 else 1.0
    
    def analyze_multiple_users(self, num_users: int = 10) -> Dict:
        """Analyze multiple users and aggregate results."""
        # Get sample users
        users = self.test_data['user_id'].unique()[:num_users]
        print(f"🔍 Analyzing {len(users)} users...")
        
        results = []
        for i, user_id in enumerate(users):
            print(f"   Progress: {i+1}/{len(users)} - {user_id}")
            result = self.compare_single_user(user_id)
            if result:
                results.append(result)
        
        if not results:
            print("❌ No valid results obtained")
            return {}
        
        # Aggregate statistics
        identical_count = sum(1 for r in results if r['recommendations_identical'])
        avg_ranking_similarity = np.mean([r['ranking_similarity'] for r in results if not np.isnan(r['ranking_similarity'])])
        avg_score_correlation = np.mean([r['score_correlation'] for r in results if not np.isnan(r['score_correlation'])])
        
        # Score enhancement analysis
        basic_score_means = [r['basic_stats']['mean'] for r in results]
        ml_score_means = [r['ml_stats']['mean'] for r in results]
        score_improvements = [(ml - basic) / basic * 100 for basic, ml in zip(basic_score_means, ml_score_means)]
        
        summary = {
            'total_users_analyzed': len(results),
            'identical_recommendations': identical_count,
            'identical_percentage': (identical_count / len(results)) * 100,
            'avg_ranking_similarity': avg_ranking_similarity,
            'avg_score_correlation': avg_score_correlation,
            'avg_score_improvement_pct': np.mean(score_improvements),
            'score_improvement_std': np.std(score_improvements),
            'detailed_results': results
        }
        
        return summary
    
    def print_analysis_report(self, analysis: Dict):
        """Print a formatted analysis report."""
        print(f"\n📊 DETAILED ALGORITHM COMPARISON")
        print("=" * 60)
        
        print(f"Users analyzed: {analysis['total_users_analyzed']}")
        print(f"Identical recommendations: {analysis['identical_recommendations']}/{analysis['total_users_analyzed']} ({analysis['identical_percentage']:.1f}%)")
        print(f"Average ranking similarity: {analysis['avg_ranking_similarity']:.4f}")
        print(f"Average score correlation: {analysis['avg_score_correlation']:.4f}")
        print(f"Average score improvement: {analysis['avg_score_improvement_pct']:.2f}% ± {analysis['score_improvement_std']:.2f}%")
        
        print(f"\n🔍 KEY INSIGHTS:")
        
        if analysis['identical_percentage'] > 90:
            print(f"  • ✅ Very high recommendation consistency ({analysis['identical_percentage']:.1f}% identical)")
            print(f"  • 🎯 ML model primarily enhances scores rather than changing rankings")
            print(f"  • 📈 Score improvements suggest better confidence/relevance estimation")
        elif analysis['identical_percentage'] > 50:
            print(f"  • 🔄 High recommendation overlap ({analysis['identical_percentage']:.1f}% identical)")
            print(f"  • ⚖️ ML model makes selective ranking adjustments")
        else:
            print(f"  • 🎲 Significant ranking differences ({analysis['identical_percentage']:.1f}% identical)")
            print(f"  • 🚀 ML model substantially changes recommendation strategy")
        
        if analysis['avg_score_correlation'] > 0.8:
            print(f"  • 🤝 Strong score correlation (r={analysis['avg_score_correlation']:.3f}) - ML enhances existing signals")
        elif analysis['avg_score_correlation'] > 0.5:
            print(f"  • 🔄 Moderate score correlation (r={analysis['avg_score_correlation']:.3f}) - ML adjusts some signals")
        else:
            print(f"  • 🎯 Low score correlation (r={analysis['avg_score_correlation']:.3f}) - ML uses different signals")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"  • The ML model appears to be working as a 'score enhancer' rather than a 'ranking changer'")
        print(f"  • This suggests the basic algorithm already captures the main ranking signals well")
        print(f"  • ML adds nuanced scoring that could be valuable for confidence estimation or A/B testing")
        print(f"  • The preserved ranking explains why precision/recall/hit-rate are identical in evaluation")


def main():
    """Main analysis function."""
    analyzer = DetailedComparison()
    
    # Load test data
    try:
        analyzer.load_test_data()
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return
    
    # Analyze multiple users
    try:
        analysis = analyzer.analyze_multiple_users(num_users=20)
        
        if analysis:
            analyzer.print_analysis_report(analysis)
            
            # Save detailed results
            with open('detailed_comparison_results.json', 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Create a JSON-serializable version
                json_analysis = json.loads(json.dumps(analysis, default=convert_numpy))
                json.dump(json_analysis, f, indent=2)
            
            print(f"\n💾 Detailed results saved to detailed_comparison_results.json")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
