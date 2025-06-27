#!/usr/bin/env python3
"""
Evaluation script for HOMETOWN recommendation algorithms.

This script loads the test set and evaluates both the basic and ML-enhanced 
recommendation APIs using standard ranking metrics like nDCG@5, Precision@5, etc.
"""

import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import time


class RecommendationEvaluator:
    """Evaluates recommendation APIs using standard ranking metrics."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_data = None
        self.user_ground_truth = {}
        
    def load_test_data(self, test_file: str = "data/hometown_test.parquet"):
        """Load test data and prepare ground truth."""
        print("📊 Loading test data...")
        self.test_data = pd.read_parquet(test_file)
        
        # Create ground truth: for each user, list of streams they clicked on
        for _, row in self.test_data.iterrows():
            user_id = row['user_id']
            stream_id = row['stream_id']
            clicked = row['clicked']
            
            if user_id not in self.user_ground_truth:
                self.user_ground_truth[user_id] = {
                    'clicked_streams': set(),
                    'all_streams': set()
                }
            
            self.user_ground_truth[user_id]['all_streams'].add(stream_id)
            if clicked == 1:
                self.user_ground_truth[user_id]['clicked_streams'].add(stream_id)
        
        print(f"✅ Loaded {len(self.test_data)} interactions for {len(self.user_ground_truth)} users")
        print(f"📈 Total positive interactions: {self.test_data['clicked'].sum()}")
        
        # Count users with positive interactions for evaluation
        users_with_clicks = sum(1 for user_data in self.user_ground_truth.values() 
                               if len(user_data['clicked_streams']) > 0)
        users_without_clicks = len(self.user_ground_truth) - users_with_clicks
        
        print(f"👥 Users with positive interactions: {users_with_clicks}")
        print(f"🚫 Users with only negative interactions: {users_without_clicks}")
        print(f"ℹ️  Note: Only users with positive interactions can be evaluated for ranking quality")
        
    def get_recommendations(self, user_id: str, endpoint: str, k: int = 10) -> List[str]:
        """Get recommendations from API endpoint."""
        url = f"{self.base_url}/v1/scenarios/{endpoint}"
        payload = {"user_id": user_id, "max_results": k}
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [stream['stream_id'] for stream in data.get('streams', [])]
            else:
                print(f"⚠️ API error for {user_id} on {endpoint}: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Request failed for {user_id} on {endpoint}: {e}")
            return []
    
    def calculate_dcg(self, relevance_scores: List[int], k: int = None) -> float:
        """Calculate Discounted Cumulative Gain."""
        if k:
            relevance_scores = relevance_scores[:k]
        
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    def calculate_ndcg(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        if not relevant:
            return 0.0
        
        # Relevance scores for recommended items (1 if relevant, 0 if not)
        relevance_scores = [1 if item in relevant else 0 for item in recommended[:k]]
        
        # Calculate DCG
        dcg = self.calculate_dcg(relevance_scores, k)
        
        # Calculate Ideal DCG (best possible ranking)
        ideal_relevance = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
        idcg = self.calculate_dcg(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_precision_at_k(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate Precision at k."""
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_in_k = len([item for item in recommended_k if item in relevant])
        return relevant_in_k / k
    
    def calculate_recall_at_k(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate Recall at k."""
        if not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_in_k = len([item for item in recommended_k if item in relevant])
        return relevant_in_k / len(relevant)
    
    def calculate_hit_rate_at_k(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate Hit Rate at k (binary: did we hit any relevant item?)."""
        if not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        return 1.0 if any(item in relevant for item in recommended_k) else 0.0
    
    def calculate_f1_at_k(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate F1-score at k."""
        precision = self.calculate_precision_at_k(recommended, relevant, k)
        recall = self.calculate_recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_map_at_k(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate Mean Average Precision at k."""
        if not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        ap = 0.0
        relevant_count = 0
        
        for i, item in enumerate(recommended_k):
            if item in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        
        return ap / min(len(relevant), k) if relevant else 0.0
    
    def calculate_mrr(self, recommended: List[str], relevant: set) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not relevant:
            return 0.0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_coverage_at_k(self, recommended: List[str], relevant: set, k: int = 5) -> float:
        """Calculate Coverage at k (percentage of relevant items covered)."""
        if not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        covered = len([item for item in recommended_k if item in relevant])
        return covered / len(relevant)
    
    def calculate_novelty_at_k(self, recommended: List[str], all_streams: set, k: int = 5) -> float:
        """Calculate Novelty at k (1 - popularity bias)."""
        # Simple novelty measure: fraction of recommended items not in user's history
        if not all_streams:
            return 1.0
        
        recommended_k = recommended[:k]
        novel_items = [item for item in recommended_k if item not in all_streams]
        return len(novel_items) / len(recommended_k) if recommended_k else 0.0
    
    def evaluate_endpoint(self, endpoint: str, k: int = 5) -> Dict:
        """Evaluate a single API endpoint."""
        print(f"🔍 Evaluating {endpoint} endpoint...")
        
        metrics = {
            'ndcg': [],
            'precision': [],
            'recall': [],
            'hit_rate': [],
            'f1': [],
            'map': [],
            'mrr': [],
            'coverage': [],
            'novelty': [],
            'response_times': [],
            'errors': 0
        }
        
        users_to_test = list(self.user_ground_truth.keys())  # Test all users for comprehensive evaluation
        print(f"📋 Testing {len(users_to_test)} users...")
        
        for i, user_id in enumerate(users_to_test):
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{len(users_to_test)} users")
            
            # Get ground truth for this user
            relevant_streams = self.user_ground_truth[user_id]['clicked_streams']
            all_streams_user = self.user_ground_truth[user_id]['all_streams']
            
            if not relevant_streams:
                # Skip users with no positive interactions - can't evaluate ranking quality
                # without knowing what the user considers "relevant"
                continue
            
            # Get recommendations
            start_time = time.time()
            recommended_streams = self.get_recommendations(user_id, endpoint, k=10)
            response_time = time.time() - start_time
            
            if not recommended_streams:
                metrics['errors'] += 1
                continue
            
            # Calculate metrics
            ndcg = self.calculate_ndcg(recommended_streams, relevant_streams, k)
            precision = self.calculate_precision_at_k(recommended_streams, relevant_streams, k)
            recall = self.calculate_recall_at_k(recommended_streams, relevant_streams, k)
            hit_rate = self.calculate_hit_rate_at_k(recommended_streams, relevant_streams, k)
            f1 = self.calculate_f1_at_k(recommended_streams, relevant_streams, k)
            map_score = self.calculate_map_at_k(recommended_streams, relevant_streams, k)
            mrr = self.calculate_mrr(recommended_streams, relevant_streams)
            coverage = self.calculate_coverage_at_k(recommended_streams, relevant_streams, k)
            novelty = self.calculate_novelty_at_k(recommended_streams, all_streams_user, k)
            
            metrics['ndcg'].append(ndcg)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['hit_rate'].append(hit_rate)
            metrics['f1'].append(f1)
            metrics['map'].append(map_score)
            metrics['mrr'].append(mrr)
            metrics['coverage'].append(coverage)
            metrics['novelty'].append(novelty)
            metrics['response_times'].append(response_time)
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['ndcg', 'precision', 'recall', 'hit_rate', 'f1', 'map', 'mrr', 'coverage', 'novelty', 'response_times']:
            if metrics[metric]:
                avg_metrics[f'avg_{metric}'] = np.mean(metrics[metric])
                avg_metrics[f'std_{metric}'] = np.std(metrics[metric])
            else:
                avg_metrics[f'avg_{metric}'] = 0.0
                avg_metrics[f'std_{metric}'] = 0.0
        
        avg_metrics['total_users'] = len(users_to_test)
        avg_metrics['evaluated_users'] = len(metrics['ndcg'])
        avg_metrics['errors'] = metrics['errors']
        
        return avg_metrics
    
    def run_full_evaluation(self, k: int = 5):
        """Run evaluation on both endpoints and compare results."""
        print("🎯 HOMETOWN Recommendation Evaluation")
        print("=" * 50)
        
        # Check API availability
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code != 200:
                print(f"❌ API not available at {self.base_url}")
                return
            print(f"✅ API available at {self.base_url}")
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return
        
        # Evaluate both endpoints
        results = {}
        
        print(f"\n🧮 Evaluating with k={k} (top-{k} recommendations)")
        
        # Basic algorithm
        results['basic'] = self.evaluate_endpoint('hometown', k)
        
        # ML-enhanced algorithm
        results['ml'] = self.evaluate_endpoint('hometown-ml', k)
        
        # Print results
        self.print_comparison_results(results, k)
        
        return results
    
    def print_comparison_results(self, results: Dict, k: int):
        """Print formatted comparison results."""
        print(f"\n📊 EVALUATION RESULTS (Top-{k})")
        print("=" * 80)
        
        # Primary metrics
        primary_metrics = ['ndcg', 'precision', 'recall', 'hit_rate', 'f1']
        print("PRIMARY RANKING METRICS:")
        print(f"{'Metric':<15} {'Basic Algorithm':<20} {'ML-Enhanced':<20} {'Improvement':<15}")
        print("-" * 70)
        
        for metric in primary_metrics:
            basic_val = results['basic'][f'avg_{metric}']
            ml_val = results['ml'][f'avg_{metric}']
            
            if basic_val > 0:
                improvement = ((ml_val - basic_val) / basic_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<15} {basic_val:.4f}{'':<12} {ml_val:.4f}{'':<12} {improvement_str:<15}")
        
        # Secondary metrics
        secondary_metrics = ['map', 'mrr', 'coverage', 'novelty']
        print(f"\nSECONDARY METRICS:")
        print(f"{'Metric':<15} {'Basic Algorithm':<20} {'ML-Enhanced':<20} {'Improvement':<15}")
        print("-" * 70)
        
        for metric in secondary_metrics:
            basic_val = results['basic'][f'avg_{metric}']
            ml_val = results['ml'][f'avg_{metric}']
            
            if basic_val > 0:
                improvement = ((ml_val - basic_val) / basic_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            metric_name = metric.replace('_', ' ').title()
            if metric == 'map':
                metric_name = 'MAP'
            elif metric == 'mrr':
                metric_name = 'MRR'
            print(f"{metric_name:<15} {basic_val:.4f}{'':<12} {ml_val:.4f}{'':<12} {improvement_str:<15}")
        
        # Performance metrics
        print(f"\nPERFORMANCE METRICS:")
        print(f"{'Metric':<15} {'Basic Algorithm':<20} {'ML-Enhanced':<20} {'Improvement':<15}")
        print("-" * 70)
        
        basic_time = results['basic']['avg_response_times']
        ml_time = results['ml']['avg_response_times']
        time_improvement = ((ml_time - basic_time) / basic_time) * 100
        print(f"Response Time{'':<2} {basic_time*1000:.1f}ms{'':<12} {ml_time*1000:.1f}ms{'':<12} {time_improvement:+.1f}%{'':<6}")
        
        print("\n📈 DETAILED SUMMARY")
        print("-" * 50)
        print(f"Basic Algorithm:")
        print(f"  • Users evaluated: {results['basic']['evaluated_users']}/{results['basic']['total_users']}")
        print(f"  • Errors: {results['basic']['errors']}")
        print(f"  • nDCG@{k}: {results['basic']['avg_ndcg']:.4f} ± {results['basic']['std_ndcg']:.4f}")
        print(f"  • MAP@{k}: {results['basic']['avg_map']:.4f} ± {results['basic']['std_map']:.4f}")
        print(f"  • MRR: {results['basic']['avg_mrr']:.4f} ± {results['basic']['std_mrr']:.4f}")
        
        print(f"\nML-Enhanced Algorithm:")
        print(f"  • Users evaluated: {results['ml']['evaluated_users']}/{results['ml']['total_users']}")
        print(f"  • Errors: {results['ml']['errors']}")
        print(f"  • nDCG@{k}: {results['ml']['avg_ndcg']:.4f} ± {results['ml']['std_ndcg']:.4f}")
        print(f"  • MAP@{k}: {results['ml']['avg_map']:.4f} ± {results['ml']['std_map']:.4f}")
        print(f"  • MRR: {results['ml']['avg_mrr']:.4f} ± {results['ml']['std_mrr']:.4f}")
        
        # Statistical significance indicator
        ndcg_improvement = results['ml']['avg_ndcg'] - results['basic']['avg_ndcg']
        map_improvement = results['ml']['avg_map'] - results['basic']['avg_map']
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"  • nDCG@{k} improvement: {ndcg_improvement:+.4f}")
        print(f"  • MAP@{k} improvement: {map_improvement:+.4f}")
        
        if ndcg_improvement > 0.01 or map_improvement > 0.01:
            print(f"  • ✅ ML model shows meaningful improvement")
        elif ndcg_improvement > 0 or map_improvement > 0:
            print(f"  • 🔄 ML model shows slight improvement")
        else:
            print(f"  • ⚠️ ML model shows no improvement")
        
        # Coverage and diversity insights
        basic_coverage = results['basic']['avg_coverage']
        ml_coverage = results['ml']['avg_coverage']
        basic_novelty = results['basic']['avg_novelty']
        ml_novelty = results['ml']['avg_novelty']
        
        print(f"\n📊 DIVERSITY & COVERAGE:")
        print(f"  • Coverage@{k}: Basic={basic_coverage:.3f}, ML={ml_coverage:.3f}")
        print(f"  • Novelty@{k}: Basic={basic_novelty:.3f}, ML={ml_novelty:.3f}")
        
        if abs(basic_novelty - ml_novelty) > 0.01:
            novelty_change = "higher" if ml_novelty > basic_novelty else "lower"
            print(f"  • ML recommendations are {novelty_change} novelty")
        else:
            print(f"  • Similar novelty between algorithms")


def main():
    """Main evaluation function."""
    evaluator = RecommendationEvaluator()
    
    # Load test data
    try:
        evaluator.load_test_data()
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return
    
    # Run evaluation
    try:
        results = evaluator.run_full_evaluation(k=5)
        
        # Save results
        if results:
            with open('evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to evaluation_results.json")
            
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation stopped by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")


if __name__ == "__main__":
    main()
