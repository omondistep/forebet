import sys
import re
import math
import json
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from collections import namedtuple, Counter, defaultdict
from datetime import datetime, timedelta
import statistics
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
# Removed automatic scheduling imports - now using manual learning
# import sqlite3  # Removed - no longer needed for auto-verification

class MatchType(Enum):
    """Type of match for context-aware analysis"""
    LEAGUE = "league"
    CUP = "cup"
    FRIENDLY = "friendly"
    PLAYOFF = "playoff"
    UNKNOWN = "unknown"

Match = namedtuple('Match', ['date', 'home_team', 'away_team', 'score', 'result', 'goals_scored', 'goals_conceded', 'venue', 'competition', 'opponent_position'])

@dataclass
class H2HMatch:
    """Head-to-head match data"""
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    competition: str
    match_type: MatchType = MatchType.LEAGUE

@dataclass
class H2HMetrics:
    """Head-to-head metrics"""
    total_matches: int = 0
    home_wins: int = 0
    away_wins: int = 0
    draws: int = 0
    home_goals_for: float = 0.0
    home_goals_against: float = 0.0
    away_goals_for: float = 0.0
    away_goals_against: float = 0.0
    recent_results: List[str] = None
    draw_rate: float = 0.0
    home_win_rate: float = 0.0
    away_win_rate: float = 0.0
    avg_total_goals: float = 0.0
    recent_trend: float = 0.0
    home_advantage_h2h: float = 0.0
    goal_trend_h2h: float = 0.0
    
    def __post_init__(self):
        if self.recent_results is None:
            self.recent_results = []

@dataclass
class PredictionOutcome:
    """Track prediction outcomes for learning"""
    match_id: str
    timestamp: datetime
    home_team: str
    away_team: str
    predicted_result: str
    actual_result: str
    predicted_probabilities: Dict[str, float]
    confidence_score: int
    key_factors_used: List[str]
    match_characteristics: List[str]
    was_correct: bool
    error_magnitude: float
    odds_at_time: Optional[Dict[str, float]] = None
    home_metrics: Optional[Dict] = None
    away_metrics: Optional[Dict] = None
    core_analysis: Optional[Dict] = None

@dataclass
class LeagueSpecificMetrics:
    league_name: str = ""
    home_advantage_factor: float = 1.0
    draw_tendency: float = 0.33
    goal_scoring_rate: float = 2.5
    upset_frequency: float = 0.15
    sample_size: int = 0

@dataclass
class LearningMetrics:
    """Track learning progress"""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    false_positive_rate: Dict[str, float] = None
    factor_effectiveness: Dict[str, float] = None
    common_misclassification_patterns: List[str] = None
    recent_accuracy_trend: float = 0.0
    model_updates: int = 0
    last_update: datetime = None
    league_metrics: Dict[str, LeagueSpecificMetrics] = None
    
    def __post_init__(self):
        if self.false_positive_rate is None:
            self.false_positive_rate = {'home_win': 0.0, 'away_win': 0.0, 'draw': 0.0}
        if self.factor_effectiveness is None:
            self.factor_effectiveness = {}
        if self.common_misclassification_patterns is None:
            self.common_misclassification_patterns = []
        if self.league_metrics is None:
            self.league_metrics = {}

# PendingPrediction removed - now using manual learning input

@dataclass 
class TeamMetrics:
    """Comprehensive team performance metrics computed from match data"""
    team_name: str
    
    # Core metrics computed from last 6-8 matches (no defaults)
    form_rating: float
    recent_form_rating: float
    goal_scoring_rate: float
    goal_conceding_rate: float
    clean_sheet_rate: float
    
    # Performance against quality opponents (no defaults)
    top6_opponents_faced: int
    performance_vs_top6: float
    top6_performance_score: float
    avg_opponent_position: float
    opponent_strength_factor: float
    
    # Match-specific indicators (no defaults)
    scoring_first_rate: float
    shots_efficiency: float
    possession_dominance: float
    home_advantage: float
    away_performance: float
    
    # League position (no default)
    league_position: Optional[int]
    
    # Recent momentum and trends (no defaults)
    momentum: float
    recent_momentum: float
    goal_trend: float
    defense_trend: float
    
    # Match consistency (no default)
    consistency_score: float
    
    # ====== FIELDS WITH DEFAULT VALUES ======
    vs_bottom_half_performance: float = 0.5
    home_scoring_rate: float = 0.0
    away_scoring_rate: float = 0.0
    home_conceding_rate: float = 0.0
    away_conceding_rate: float = 0.0
    home_away_trend: float = 0.0
    result_consistency: float = 0.5
    performance_variance: float = 0.0
    match_type_adjustment: float = 1.0
    league_tier_factor: float = 1.0
    position_trend: int = 0
    h2h_metrics: Optional[H2HMetrics] = None
    
    # New metrics based on wrong predictions analysis
    late_goal_tendency: float = 0.0
    comeback_ability: float = 0.0
    hold_lead_ability: float = 0.0
    draw_tendency: float = 0.0
    big_win_tendency: float = 0.0
    big_loss_tendency: float = 0.0
    
    # Enhanced metrics for learning
    recent_fixture_difficulty: float = 0.0
    days_since_last_match: int = 0
    injury_impact_score: float = 1.0
    manager_stability: float = 1.0
    motivation_factor: float = 1.0
    
    def __post_init__(self):
        if self.h2h_metrics is None:
            self.h2h_metrics = H2HMetrics()

@dataclass
class MatchFactors:
    """Weighted factors for match prediction - OPTIMIZED BASED ON WRONG PREDICTIONS"""
    # Reduced weight for form, increased for specific indicators
    form_weight: float = 0.15
    recent_form_weight: float = 0.14
    attack_vs_defense_weight: float = 0.20
    opponent_quality_weight: float = 0.18
    home_advantage_weight: float = 0.12
    momentum_weight: float = 0.06
    consistency_weight: float = 0.10
    h2h_weight: float = 0.16
    top6_performance_weight: float = 0.08
    tactical_mismatch_weight: float = 0.05
    comeback_ability_weight: float = 0.03
    league_context_weight: float = 0.04
    # New factors from learning
    fixture_congestion_weight: float = 0.04
    motivation_weight: float = 0.05
    injury_impact_weight: float = 0.06
    draw_tendency_weight: float = 0.05

@dataclass
class CoreAnalysis:
    """Analysis of match determinants based on actual data"""
    form_difference: float = 0.0
    recent_form_difference: float = 0.0
    goal_scoring_difference: float = 0.0
    goal_conceding_difference: float = 0.0
    opponent_quality_difference: float = 0.0
    home_advantage_strength: float = 0.0
    momentum_difference: float = 0.0
    consistency_difference: float = 0.0
    top6_performance_difference: float = 0.0
    h2h_dominance: float = 0.0
    tactical_mismatch: float = 0.0
    comeback_advantage: float = 0.0
    league_context_advantage: float = 0.0
    expected_goals_advantage: float = 0.0
    key_factors: List[str] = None
    detected_patterns: List[str] = None
    value_adjustments: List[float] = None
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []
        if self.detected_patterns is None:
            self.detected_patterns = []
        if self.value_adjustments is None:
            self.value_adjustments = []

@dataclass
class MatchSummary:
    """Summary of match characteristics for decision making"""
    is_close_contest: bool = False
    expected_goals_total: float = 0.0
    bts_likely: bool = False
    high_scoring_potential: bool = False
    low_scoring_potential: bool = False
    home_dominant_h2h: bool = False
    away_dominant_h2h: bool = False
    draw_tendency_h2h: bool = False
    is_mismatch: bool = False
    is_derby: bool = False
    tactical_advantage: str = ""
    is_unpredictable: bool = False
    has_momentum_shift: bool = False
    is_defensive_battle: bool = False
    is_goalfest_potential: bool = False
    home_goal_scoring_advantage: float = 0.0
    away_goal_scoring_advantage: float = 0.0
    home_defensive_advantage: float = 0.0
    away_defensive_advantage: float = 0.0
    characteristics: List[str] = None
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = []

class LearningEngine:
    """Analyze prediction outcomes and adjust weights"""
    
    def __init__(self, storage_file="prediction_history.json"):
        self.storage_file = storage_file
        self.history = self.load_history()
        self.learning_metrics = self.load_metrics()
        self.goal_learning_metrics = {
            'over25_correct': 0, 'over25_total': 0,
            'bts_correct': 0, 'bts_total': 0,
            'avg_goals_error': 0.0, 'goal_samples': 0
        }
    
    def load_history(self) -> List[PredictionOutcome]:
        """Load prediction history from file"""
        try:
            if Path(self.storage_file).exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                history = []
                for item in data:
                    # Convert string timestamp back to datetime
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    history.append(PredictionOutcome(**item))
                return history
        except:
            pass
        return []
    
    def load_metrics(self) -> LearningMetrics:
        """Load learning metrics from file"""
        try:
            metrics_file = "learning_metrics.json"
            if Path(metrics_file).exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                data['last_update'] = datetime.fromisoformat(data['last_update']) if data.get('last_update') else None
                return LearningMetrics(**data)
        except:
            pass
        return LearningMetrics()
    
    def save_history(self):
        """Save prediction history to file"""
        try:
            with open(self.storage_file, 'w') as f:
                json_data = []
                for outcome in self.history:
                    outcome_dict = asdict(outcome)
                    outcome_dict['timestamp'] = outcome.timestamp.isoformat()
                    json_data.append(outcome_dict)
                json.dump(json_data, f, indent=2)
        except:
            pass
    
    def save_metrics(self):
        """Save learning metrics to file"""
        try:
            self.learning_metrics.last_update = datetime.now()
            metrics_dict = asdict(self.learning_metrics)
            metrics_dict['last_update'] = self.learning_metrics.last_update.isoformat()
            with open("learning_metrics.json", 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        except:
            pass
    
    def record_prediction(self, outcome: PredictionOutcome):
        """Record a new prediction outcome"""
        self.history.append(outcome)
        self.learning_metrics.total_predictions += 1
        if outcome.was_correct:
            self.learning_metrics.correct_predictions += 1
        
        # Update accuracy
        self.learning_metrics.accuracy = (
            self.learning_metrics.correct_predictions / 
            max(1, self.learning_metrics.total_predictions)
        )
        
        # Track false positives by type
        if not outcome.was_correct:
            pred_type = outcome.predicted_result.lower()
            if pred_type in ['h', 'a', 'd']:
                key = {'h': 'home_win', 'a': 'away_win', 'd': 'draw'}[pred_type]
                self.learning_metrics.false_positive_rate[key] += 1
        
        # Save periodically
        if self.learning_metrics.total_predictions % 10 == 0:
            self.save_history()
            self.save_metrics()
    
    def analyze_wrong_prediction(self, outcome: PredictionOutcome) -> List[str]:
        """Analyze why a prediction was wrong and extract lessons"""
        
        lessons = []
        
        # Check form-related errors
        if (outcome.predicted_result == 'H' and 
            outcome.core_analysis and 
            outcome.core_analysis.get('form_difference', 0) > 0.2 and 
            outcome.actual_result != 'H'):
            lessons.append("Form difference overvalued - team underperformed vs expectation")
        
        # Check H2H pattern breaks
        if (outcome.home_metrics and 
            outcome.home_metrics.get('h2h_metrics', {}).get('total_matches', 0) >= 3 and
            outcome.actual_result == 'D'):
            lessons.append("Strong H2H pattern broken by draw")
        
        # Check momentum errors
        if (outcome.core_analysis and 
            abs(outcome.core_analysis.get('momentum_difference', 0)) > 0.4 and 
            outcome.actual_result != outcome.predicted_result):
            lessons.append("Momentum shift didn't translate to result")
        
        # Check tactical mismatch failures
        if (outcome.core_analysis and 
            abs(outcome.core_analysis.get('tactical_mismatch', 0)) > 0.2 and 
            outcome.actual_result != outcome.predicted_result):
            lessons.append("Tactical advantage didn't materialize")
        
        # Check consistency failures
        if (outcome.core_analysis and 
            abs(outcome.core_analysis.get('consistency_difference', 0)) > 0.3 and 
            outcome.actual_result != outcome.predicted_result):
            if outcome.core_analysis.get('consistency_difference', 0) > 0:
                lessons.append("More consistent team underperformed")
            else:
                lessons.append("Less consistent team overperformed")
        
        # Add to common patterns if seen multiple times
        if lessons:
            for lesson in lessons:
                if lesson not in self.learning_metrics.common_misclassification_patterns:
                    self.learning_metrics.common_misclassification_patterns.append(lesson)
        
        return lessons
    
    def record_goal_outcome(self, predicted_over25: bool, actual_over25: bool,
                          predicted_bts: bool, actual_bts: bool,
                          total_goals: int, factors: MatchFactors):
        """Record goal prediction outcomes for learning"""
        
        # Over 2.5 goals learning
        self.goal_learning_metrics['over25_total'] += 1
        if predicted_over25 == actual_over25:
            self.goal_learning_metrics['over25_correct'] += 1
        
        # Both teams to score learning
        self.goal_learning_metrics['bts_total'] += 1
        if predicted_bts == actual_bts:
            self.goal_learning_metrics['bts_correct'] += 1
        
        # Goal prediction accuracy tracking
        self.goal_learning_metrics['goal_samples'] += 1

    def adjust_factor_weights(self, current_factors: MatchFactors) -> MatchFactors:
        """Dynamically adjust factor weights based on prediction errors"""
        
        # CRITICAL LEARNING-BASED ADJUSTMENTS (From 33+ match analysis)
        if self.learning_metrics.total_predictions > 20:
            accuracy = self.learning_metrics.correct_predictions / self.learning_metrics.total_predictions
            
            # Major adjustments for poor performance (current accuracy ~30%)
            if accuracy < 0.4:
                # Fix away bias - model predicted 24 away wins, only 5 occurred
                current_factors.attack_vs_defense_weight = max(0.18, current_factors.attack_vs_defense_weight * 0.85)
                # Increase home advantage - was undervalued
                current_factors.home_advantage_weight = min(0.20, current_factors.home_advantage_weight * 1.4)
                # Boost H2H for draw detection - 13 actual draws, 0 predicted
                current_factors.h2h_weight = min(0.22, current_factors.h2h_weight * 1.3)
                current_factors.consistency_weight = min(0.15, current_factors.consistency_weight * 1.2)
        
        if len(self.history) < 20:
            return current_factors
        
        # Get recent wrong predictions
        recent_predictions = self.history[-50:]  # Last 50 predictions
        wrong_predictions = [p for p in recent_predictions if not p.was_correct]
        
        if len(wrong_predictions) < 10:
            return current_factors
        
        # Analyze patterns
        pattern_analysis = defaultdict(int)
        total_errors = len(wrong_predictions)
        
        for outcome in wrong_predictions:
            lessons = self.analyze_wrong_prediction(outcome)
            for lesson in lessons:
                pattern_analysis[lesson] += 1
        
        # Calculate adjustments
        adjustments = {}
        
        # Form weight adjustments
        form_errors = sum(1 for pattern in pattern_analysis.keys() 
                         if "Form difference" in pattern or "form" in pattern.lower())
        if form_errors / total_errors > 0.3:
            adjustments['form_weight'] = -0.015
            adjustments['recent_form_weight'] = -0.01
        
        # H2H weight adjustments
        h2h_errors = sum(1 for pattern in pattern_analysis.keys() 
                        if "H2H" in pattern)
        if h2h_errors / total_errors > 0.25:
            adjustments['h2h_weight'] = -0.02
        
        # Momentum weight adjustments
        momentum_errors = sum(1 for pattern in pattern_analysis.keys() 
                            if "Momentum" in pattern)
        if momentum_errors / total_errors > 0.2:
            adjustments['momentum_weight'] = -0.01
        
        # Consistency weight adjustments
        consistency_errors = sum(1 for pattern in pattern_analysis.keys() 
                               if "consistent" in pattern.lower())
        if consistency_errors / total_errors > 0.15:
            adjustments['consistency_weight'] = +0.015
        
        # Draw tendency adjustments
        draw_errors = sum(1 for outcome in wrong_predictions 
                         if outcome.predicted_result != 'D' and outcome.actual_result == 'D')
        if draw_errors / total_errors > 0.2:
            adjustments['draw_tendency_weight'] = +0.02
        
        # Apply adjustments with limits
        new_factors = MatchFactors()
        for attr, adjustment in adjustments.items():
            current_val = getattr(new_factors, attr)
            new_val = max(0.03, min(0.30, current_val + adjustment))
            setattr(new_factors, attr, new_val)
        
        # Record model update
        self.learning_metrics.model_updates += 1
        self.save_metrics()
        
        return new_factors
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            'total_predictions': self.learning_metrics.total_predictions,
            'accuracy': f"{self.learning_metrics.accuracy:.1%}",
            'recent_accuracy': self.calculate_recent_accuracy(50),
            'model_updates': self.learning_metrics.model_updates,
            'common_errors': self.learning_metrics.common_misclassification_patterns[:5],
            'false_positive_rates': self.learning_metrics.false_positive_rate
        }
    
    def calculate_recent_accuracy(self, n: int) -> float:
        """Calculate accuracy for last n predictions"""
        if len(self.history) < n:
            return 0.0
        recent = self.history[-n:]
        correct = sum(1 for p in recent if p.was_correct)
        return correct / len(recent)

class PatternRecognizer:
    """Recognize specific match patterns that lead to upsets"""
    
    UPSET_PATTERNS = {
        'strong_home_team_slump': {
            'conditions': [
                lambda hm, am: hm.form_rating > 0.7,
                lambda hm, am: hm.recent_form_rating < 0.4,
                lambda hm, am: am.away_performance > 0.6
            ],
            'adjustment': -0.15,
            'message': "Strong home team in recent slump vs good away performer"
        },
        'false_favorite': {
            'conditions': [
                lambda hm, am: hm.avg_opponent_position > 14,
                lambda hm, am: hm.form_rating > 0.65,
                lambda hm, am: am.avg_opponent_position < 10
            ],
            'adjustment': -0.10,
            'message': "Home team faced weak opponents, away team faced strong ones"
        },
        'draw_specialists_clash': {
            'conditions': [
                lambda hm, am: hm.draw_tendency > 0.4,
                lambda hm, am: am.draw_tendency > 0.4,
                lambda hm, am: abs(hm.form_rating - am.form_rating) < 0.2
            ],
            'adjustment': +0.25,
            'message': "Both teams are draw specialists with similar form"
        },
        'momentum_reversal': {
            'conditions': [
                lambda hm, am: hm.recent_momentum < -0.5,
                lambda hm, am: am.recent_momentum > 0.5,
                lambda hm, am: abs(hm.form_rating - am.form_rating) < 0.3
            ],
            'adjustment': -0.20,
            'message': "Strong negative momentum vs strong positive momentum"
        }
    }
    
    def detect_patterns(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics) -> Tuple[List[str], List[float], List[str]]:
        """Detect if any upset patterns apply"""
        detected = []
        adjustments = []
        messages = []
        
        for pattern_name, pattern in self.UPSET_PATTERNS.items():
            conditions_met = all(
                condition(home_metrics, away_metrics)
                for condition in pattern['conditions']
            )
            
            if conditions_met:
                detected.append(pattern_name)
                adjustments.append(pattern['adjustment'])
                messages.append(pattern['message'])
        
        return detected, adjustments, messages

# AutoVerificationSystem removed - now using manual learning input

class ValueBetDetector:
    """Detect value bets by comparing predictions to market odds"""
    
    def calculate_value(self, predicted_prob: float, market_odds: float) -> float:
        """Calculate value percentage"""
        implied_prob = 1 / market_odds
        return predicted_prob - implied_prob
    
    def find_value_bets(self, predictions: Dict, market_odds: Dict) -> List[Dict]:
        """Find bets with positive expected value"""
        value_bets = []
        
        # Map prediction keys to market odds keys
        prediction_to_market = {
            'home_win': '1',
            'draw': 'X',
            'away_win': '2'
        }
        
        for pred_key, pred_prob in predictions.items():
            if pred_key in ['home_win', 'away_win', 'draw']:
                market_key = prediction_to_market.get(pred_key)
                if market_key in market_odds:
                    value = self.calculate_value(pred_prob/100, market_odds[market_key])
                    if value > 0.05:  # 5% value threshold
                        value_bets.append({
                            'outcome': pred_key,
                            'predicted_prob': pred_prob,
                            'market_odds': market_odds[market_key],
                            'implied_prob': 1/market_odds[market_key] * 100,
                            'value_percentage': value * 100,
                            'expected_value': (market_odds[market_key] - 1) * (pred_prob/100) - (1 - pred_prob/100),
                            'reasoning': f'{pred_key} probability ({pred_prob:.1f}%) significantly higher than implied by market odds ({1/market_odds[market_key]*100:.1f}%)'
                        })
        
        return sorted(value_bets, key=lambda x: x['value_percentage'], reverse=True)

class AdaptiveFactors:
    """Dynamically adjust weights based on match context"""
    
    @staticmethod
    def get_contextual_factors(match_summary: MatchSummary, 
                              home_metrics: TeamMetrics, 
                              away_metrics: TeamMetrics) -> MatchFactors:
        
        base_factors = MatchFactors()
        
        # ADJUSTMENTS BASED ON MATCH CONTEXT:
        
        # 1. For derbies: Reduce form weight, increase H2H
        if match_summary.is_derby:
            base_factors.form_weight *= 0.7
            base_factors.recent_form_weight *= 0.8
            base_factors.h2h_weight *= 1.4
            base_factors.momentum_weight *= 1.2
        
        # 2. For mismatches: Increase attack/defense analysis
        if match_summary.is_mismatch:
            base_factors.attack_vs_defense_weight *= 1.3
            base_factors.form_weight *= 0.9
            base_factors.consistency_weight *= 1.1
        
        # 3. For unpredictable matches: More conservative weights
        if match_summary.is_unpredictable:
            base_factors.form_weight *= 0.6
            base_factors.recent_form_weight *= 0.7
            base_factors.consistency_weight *= 1.3
            base_factors.draw_tendency_weight *= 1.5
        
        # 4. For high-scoring potential: Different weight distribution
        if match_summary.is_goalfest_potential:
            base_factors.attack_vs_defense_weight *= 1.2
            base_factors.opponent_quality_weight *= 0.9
        
        # 5. For defensive battles: Different emphasis
        if match_summary.is_defensive_battle:
            base_factors.attack_vs_defense_weight *= 0.8
            base_factors.consistency_weight *= 1.2
            base_factors.draw_tendency_weight *= 1.3
        
        return base_factors

class DecisionEngine:
    """Engine to make betting decisions based on computed metrics"""
    
    def __init__(self):
        self.learning_engine = LearningEngine()
        self.pattern_recognizer = PatternRecognizer()
        self.value_detector = ValueBetDetector()
        
    def detect_league(self, competition: str, home_team: str, away_team: str) -> str:
        """Detect league from competition name and team names"""
        comp_lower = competition.lower()
        
        # Champions League detection
        if any(keyword in comp_lower for keyword in ['champions league', 'ucl', 'uefa champions']):
            return "Champions League"
        
        # Conference League detection
        if any(keyword in comp_lower for keyword in ['conference league', 'uecl', 'uefa conference']):
            return "Conference League"
        
        # Europa League detection  
        if any(keyword in comp_lower for keyword in ['europa league', 'uel', 'uefa europa']):
            return "Europa League"
        
        # Premier League detection
        if any(keyword in comp_lower for keyword in ['premier league', 'epl', 'england']):
            return "Premier League"
        
        # Other major leagues
        if any(keyword in comp_lower for keyword in ['la liga', 'spain', 'laliga']):
            return "La Liga"
        if any(keyword in comp_lower for keyword in ['bundesliga', 'germany']):
            return "Bundesliga"
        if any(keyword in comp_lower for keyword in ['serie a', 'italy']):
            return "Serie A"
        if any(keyword in comp_lower for keyword in ['ligue 1', 'france']):
            return "Ligue 1"
        
        return competition or "Unknown"

    def get_league_adjustments(self, league: str) -> Dict[str, float]:
        """Get league-specific factor adjustments"""
        
        # Conference League characteristics (UPDATED - 60% draw rate, very low scoring)
        if league == "Conference League":
            return {
                'home_advantage_multiplier': 1.08,  # Moderate home advantage
                'draw_rate_expected': 0.45,         # High draw rate (60% observed)
                'goals_per_game': 1.5,              # Very low scoring (1.0 observed)
                'upset_frequency': 0.50             # High upset rate
            }
        
        # Europa League characteristics (FINAL ADJUSTMENT - 89% draw rate observed!)
        if league == "Europa League":
            return {
                'home_advantage_multiplier': 1.01,  # Almost no home advantage
                'draw_rate_expected': 0.70,         # EXTREME draw rate (89% observed!)
                'goals_per_game': 2.2,              # Moderate scoring but drawn games
                'upset_frequency': 0.50             # Extreme unpredictability
            }
        
        # Champions League characteristics (UPDATED based on learning)
        if league == "Champions League":
            return {
                'home_advantage_multiplier': 1.02,  # Very reduced home advantage
                'draw_rate_expected': 0.30,         # Higher draw rate (30% observed)
                'goals_per_game': 2.6,              # Lower scoring than expected
                'upset_frequency': 0.30             # Higher upset rate
            }
        
        # Premier League characteristics
        if league == "Premier League":
            return {
                'home_advantage_multiplier': 1.15,  # Strong home advantage
                'draw_rate_expected': 0.28,         # ~28% draw rate
                'goals_per_game': 2.8,              # High scoring
                'upset_frequency': 0.25             # High upset rate
            }
        
        return {
            'home_advantage_multiplier': 1.0,
            'draw_rate_expected': 0.33,
            'goals_per_game': 2.5,
            'upset_frequency': 0.15
        }
    
    def calculate_football_consistency(self, matches: List[Match], match_results: List[str]) -> float:
        """Calculate consistency based on football logic, not just statistical variance"""
        
        if len(matches) < 3:
            return 0.5
        
        # Count results
        wins = sum(1 for r in match_results if r == 'W')
        draws = sum(1 for r in match_results if r == 'D')
        losses = sum(1 for r in match_results if r == 'L')
        total = len(matches)
        
        win_rate = wins / total
        draw_rate = draws / total
        loss_rate = losses / total
        
        # Pattern 1: Dominant team (wins most matches)
        if win_rate >= 0.67:
            consistency = 0.8 + (win_rate * 0.2)
            return min(1.0, consistency)
        
        # Pattern 2: Always losing
        elif loss_rate >= 0.67:
            consistency = 0.7 + (loss_rate * 0.2)
            return min(0.9, consistency)
        
        # Pattern 3: Draw specialist
        elif draw_rate >= 0.5:
            consistency = 0.75 + (draw_rate * 0.15)
            return min(0.9, consistency)
        
        # Pattern 4: Mixed results - check for streaks
        else:
            current_streak = 1
            max_streak = 1
            for i in range(1, len(match_results)):
                if match_results[i] == match_results[i-1]:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
            
            streak_factor = min(1.0, max_streak / 4)
            
            if max(wins, draws, losses) >= total / 2:
                dominant_rate = max(win_rate, draw_rate, loss_rate)
                consistency = 0.6 + (dominant_rate * 0.3)
            else:
                consistency = 0.4
            
            consistency = consistency * 0.7 + streak_factor * 0.3
            
            return max(0.3, min(1.0, consistency))
    
    def calculate_home_away_specific_metrics(self, team_name: str, home_matches: List[Match], away_matches: List[Match]) -> Dict[str, float]:
        """Calculate home and away specific performance metrics"""
        
        metrics = {
            'home_form_rating': 0.5,
            'away_form_rating': 0.5,
            'home_scoring_rate': 1.0,
            'away_scoring_rate': 1.0,
            'home_conceding_rate': 1.0,
            'away_conceding_rate': 1.0,
            'home_clean_sheet_rate': 0.3,
            'away_clean_sheet_rate': 0.3,
            'home_vs_top6_performance': 0.5,
            'away_vs_top6_performance': 0.5,
            'home_momentum': 0.0,
            'away_momentum': 0.0
        }
        
        # Calculate home metrics
        if home_matches:
            home_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in home_matches)
            max_home_points = len(home_matches) * 3
            metrics['home_form_rating'] = home_points / max_home_points if max_home_points > 0 else 0.5
            
            metrics['home_scoring_rate'] = statistics.mean([m.goals_scored for m in home_matches]) if home_matches else 1.0
            metrics['home_conceding_rate'] = statistics.mean([m.goals_conceded for m in home_matches]) if home_matches else 1.0
            
            home_clean_sheets = sum(1 for m in home_matches if m.goals_conceded == 0)
            metrics['home_clean_sheet_rate'] = home_clean_sheets / len(home_matches) if home_matches else 0.3
            
            # Home vs top6 performance
            home_matches_vs_top6 = [m for m in home_matches if m.opponent_position and m.opponent_position <= 6]
            if home_matches_vs_top6:
                points_vs_top6 = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in home_matches_vs_top6)
                max_points_vs_top6 = len(home_matches_vs_top6) * 3
                metrics['home_vs_top6_performance'] = points_vs_top6 / max_points_vs_top6 if max_points_vs_top6 > 0 else 0.5
            
            # Home momentum (last 3 home matches)
            if len(home_matches) >= 3:
                last_3_home = home_matches[:3]
                previous_3_home = home_matches[3:6] if len(home_matches) >= 6 else []
                
                if previous_3_home:
                    last_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in last_3_home)
                    previous_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in previous_3_home)
                    max_points = 9
                    metrics['home_momentum'] = (last_3_points - previous_3_points) / max_points
        
        # Calculate away metrics
        if away_matches:
            away_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in away_matches)
            max_away_points = len(away_matches) * 3
            metrics['away_form_rating'] = away_points / max_away_points if max_away_points > 0 else 0.5
            
            metrics['away_scoring_rate'] = statistics.mean([m.goals_scored for m in away_matches]) if away_matches else 1.0
            metrics['away_conceding_rate'] = statistics.mean([m.goals_conceded for m in away_matches]) if away_matches else 1.0
            
            away_clean_sheets = sum(1 for m in away_matches if m.goals_conceded == 0)
            metrics['away_clean_sheet_rate'] = away_clean_sheets / len(away_matches) if away_matches else 0.3
            
            # Away vs top6 performance
            away_matches_vs_top6 = [m for m in away_matches if m.opponent_position and m.opponent_position <= 6]
            if away_matches_vs_top6:
                points_vs_top6 = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in away_matches_vs_top6)
                max_points_vs_top6 = len(away_matches_vs_top6) * 3
                metrics['away_vs_top6_performance'] = points_vs_top6 / max_points_vs_top6 if max_points_vs_top6 > 0 else 0.5
            
            # Away momentum (last 3 away matches)
            if len(away_matches) >= 3:
                last_3_away = away_matches[:3]
                previous_3_away = away_matches[3:6] if len(away_matches) >= 6 else []
                
                if previous_3_away:
                    last_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in last_3_away)
                    previous_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in previous_3_away)
                    max_points = 9
                    metrics['away_momentum'] = (last_3_points - previous_3_points) / max_points
        
        # Add W-D-L counts for display
        metrics.update({
            'home_wins': sum(1 for m in home_matches if m.result == 'W'),
            'home_draws': sum(1 for m in home_matches if m.result == 'D'),
            'home_losses': sum(1 for m in home_matches if m.result == 'L'),
            'away_wins': sum(1 for m in away_matches if m.result == 'W'),
            'away_draws': sum(1 for m in away_matches if m.result == 'D'),
            'away_losses': sum(1 for m in away_matches if m.result == 'L'),
            'home_goals_avg': metrics.get('home_scoring_rate', 1.0),
            'away_goals_avg': metrics.get('away_scoring_rate', 1.0),
            'home_matches_count': len(home_matches),
            'away_matches_count': len(away_matches)
        })
        
        return metrics
    
    def compute_metrics_from_matches(self, team_name: str, matches: List[Match], 
                                    is_home_team: bool, league_position: Optional[int],
                                    h2h_metrics: Optional[H2HMetrics] = None,
                                    match_type: MatchType = MatchType.LEAGUE,
                                    home_specific_matches: List[Match] = None,
                                    away_specific_matches: List[Match] = None) -> TeamMetrics:
        """Compute all metrics from actual match data - ENHANCED WITH HOME/AWAY ANALYSIS"""
        
        if not matches:
            # Try to use home/away specific matches if available
            if home_specific_matches or away_specific_matches:
                home_matches = home_specific_matches or []
                away_matches = away_specific_matches or []
                matches = home_matches + away_matches
        
        if not matches:
            return TeamMetrics(
                team_name=team_name,
                form_rating=0.5,
                recent_form_rating=0.5,
                goal_scoring_rate=1.0,
                goal_conceding_rate=1.0,
                clean_sheet_rate=0.3,
                top6_opponents_faced=0,
                performance_vs_top6=0.5,
                top6_performance_score=5.0,
                avg_opponent_position=10.0,
                opponent_strength_factor=0.5,
                scoring_first_rate=0.5,
                shots_efficiency=0.15,
                possession_dominance=0.5,
                home_advantage=0.0,
                away_performance=0.5,
                league_position=league_position,
                momentum=0.0,
                recent_momentum=0.0,
                goal_trend=0.0,
                defense_trend=0.0,
                consistency_score=0.5,
                vs_bottom_half_performance=0.5
            )
        
        # Use last 6 matches for form calculation, all matches for other stats
        form_matches = matches[:6] if len(matches) >= 6 else matches
        total_matches = len(matches)
        recent_matches = min(3, len(form_matches))
        
        # Points calculation - UPDATED: Quality-adjusted form based on last 6 matches only
        points = 0
        recent_points = 0
        match_results = []
        quality_adjusted_points = 0
        quality_adjusted_recent_points = 0
        
        # Use only last 6 matches for form calculation
        form_matches = matches[:6] if len(matches) >= 6 else matches
        recent_matches = min(3, len(form_matches))
        
        for i, match in enumerate(form_matches):
            match_points = 3 if match.result == 'W' else 1 if match.result == 'D' else 0
            points += match_points
            match_results.append(match.result)
            
            # Quality adjustment based on opponent position
            if match.opponent_position:
                # Stronger opponents (lower position numbers) = higher quality factor
                # Position 1-6: quality factor 1.2-1.5 (boost points against strong teams)
                # Position 7-12: quality factor 1.0 (neutral)
                # Position 13-20: quality factor 0.7-0.9 (reduce points against weak teams)
                if match.opponent_position <= 6:
                    quality_factor = 1.2 + (6 - match.opponent_position) * 0.05  # 1.2 to 1.45
                elif match.opponent_position <= 12:
                    quality_factor = 1.0
                else:
                    quality_factor = 0.9 - (match.opponent_position - 13) * 0.03  # 0.9 to 0.69
                
                quality_adjusted_points += match_points * quality_factor
            else:
                quality_adjusted_points += match_points  # No adjustment if position unknown
            
            if i < recent_matches:
                recent_points += match_points
                if match.opponent_position:
                    quality_adjusted_recent_points += match_points * quality_factor
                else:
                    quality_adjusted_recent_points += match_points
        
        # Calculate form ratings using quality-adjusted points
        max_points = len(form_matches) * 3
        max_quality_points = len(form_matches) * 3 * 1.45  # Maximum possible with strongest opponents
        
        # Standard form (for compatibility)
        form_rating = points / max_points if max_points > 0 else 0.5
        recent_form_rating = recent_points / (recent_matches * 3) if recent_matches > 0 else 0.5
        
        # Quality-adjusted form (primary metric)
        quality_form_rating = min(1.0, quality_adjusted_points / max_points) if max_points > 0 else 0.5
        quality_recent_form_rating = min(1.0, quality_adjusted_recent_points / (recent_matches * 3)) if recent_matches > 0 else 0.5
        
        # Use quality-adjusted ratings as primary form metrics
        form_rating = quality_form_rating
        recent_form_rating = quality_recent_form_rating
        
        # Debug: Show quality adjustment impact
        if len(form_matches) > 0:
            standard_form = points / (len(form_matches) * 3)
            print(f"📊 {team_name[:15]} Form: Standard {standard_form:.1%} → Quality-Adjusted {form_rating:.1%}")
        
        # Goals statistics with home/away breakdown
        goals_scored = []
        goals_conceded = []
        home_goals_scored = []
        away_goals_scored = []
        home_goals_conceded = []
        away_goals_conceded = []
        
        for m in matches:
            goals_scored.append(m.goals_scored)
            goals_conceded.append(m.goals_conceded)
            
            if m.venue == 'H':
                home_goals_scored.append(m.goals_scored)
                home_goals_conceded.append(m.goals_conceded)
            else:
                away_goals_scored.append(m.goals_scored)
                away_goals_conceded.append(m.goals_conceded)
        
        goal_scoring_rate = sum(goals_scored) / total_matches
        goal_conceding_rate = sum(goals_conceded) / total_matches
        
        # Home/away specific rates
        home_scoring_rate = statistics.mean(home_goals_scored) if home_goals_scored else goal_scoring_rate
        away_scoring_rate = statistics.mean(away_goals_scored) if away_goals_scored else goal_scoring_rate
        home_conceding_rate = statistics.mean(home_goals_conceded) if home_goals_conceded else goal_conceding_rate
        away_conceding_rate = statistics.mean(away_goals_conceded) if away_goals_conceded else goal_conceding_rate
        
        # Calculate home/away specific metrics if provided
        if home_specific_matches is not None and away_specific_matches is not None:
            home_away_metrics = self.calculate_home_away_specific_metrics(
                team_name, home_specific_matches, away_specific_matches
            )
            
            # Update home/away specific rates with more accurate data
            home_scoring_rate = home_away_metrics['home_scoring_rate']
            away_scoring_rate = home_away_metrics['away_scoring_rate']
            home_conceding_rate = home_away_metrics['home_conceding_rate']
            away_conceding_rate = home_away_metrics['away_conceding_rate']
        
        # Clean sheet rate
        clean_sheets = sum(1 for m in matches if m.goals_conceded == 0)
        clean_sheet_rate = clean_sheets / total_matches if total_matches > 0 else 0.3
        
        # Performance against quality opponents
        top6_opponents = 0
        points_vs_top6 = 0
        matches_vs_top6 = 0
        points_vs_bottom_half = 0
        matches_vs_bottom_half = 0
        opponent_positions = []
        
        for match in matches:
            if match.opponent_position:
                opponent_positions.append(match.opponent_position)
                
                if match.opponent_position <= 6:
                    top6_opponents += 1
                    matches_vs_top6 += 1
                    if match.result == 'W':
                        points_vs_top6 += 3
                    elif match.result == 'D':
                        points_vs_top6 += 1
                
                if match.opponent_position >= 11:
                    matches_vs_bottom_half += 1
                    if match.result == 'W':
                        points_vs_bottom_half += 3
                    elif match.result == 'D':
                        points_vs_bottom_half += 1
        
        avg_opponent_position = statistics.mean(opponent_positions) if opponent_positions else 10.0
        
        # Calculate opponent strength factor
        if opponent_positions:
            max_position = max(opponent_positions)
            min_position = min(opponent_positions)
            if max_position > min_position:
                normalized_positions = [(max_position - pos) / (max_position - min_position) for pos in opponent_positions]
                opponent_strength_factor = statistics.mean(normalized_positions)
            else:
                opponent_strength_factor = 0.5
        else:
            opponent_strength_factor = 0.5
        
        performance_vs_top6 = points_vs_top6 / (max(1, matches_vs_top6) * 3)
        vs_bottom_half_performance = points_vs_bottom_half / (max(1, matches_vs_bottom_half) * 3) if matches_vs_bottom_half > 0 else 0.5
        
        # Calculate top6 performance score
        if matches_vs_top6 > 0:
            base_score = (points_vs_top6 / matches_vs_top6) / 3 * 10
            
            goals_scored_vs_top6 = sum(m.goals_scored for m in matches if m.opponent_position and m.opponent_position <= 6)
            goals_conceded_vs_top6 = sum(m.goals_conceded for m in matches if m.opponent_position and m.opponent_position <= 6)
            
            goal_diff_per_match = (goals_scored_vs_top6 - goals_conceded_vs_top6) / matches_vs_top6
            
            if goal_diff_per_match > 0:
                goal_diff_adjustment = goal_diff_per_match * 0.8
            else:
                goal_diff_adjustment = goal_diff_per_match * 0.6
            
            away_wins_vs_top6 = sum(1 for m in matches if m.opponent_position and m.opponent_position <= 6 
                                   and m.result == 'W' and m.venue == 'A')
            away_win_bonus = away_wins_vs_top6 * 0.3
            
            top6_performance_score = max(0, min(10, base_score + goal_diff_adjustment + away_win_bonus))
        else:
            top6_performance_score = 5.0
        
        # Scoring first rate
        scored_first = 0
        for match in matches:
            if match.goals_scored > 0:
                if match.result == 'W' or (match.goals_scored > match.goals_conceded and match.goals_scored > 0):
                    scored_first += 1
                elif match.result == 'D':
                    scored_first += 0.5
        
        scoring_first_rate = scored_first / total_matches if total_matches > 0 else 0.5
        
        # Shots efficiency
        if goal_scoring_rate > 2.5:
            shots_efficiency = 0.30
        elif goal_scoring_rate > 2.0:
            shots_efficiency = 0.25
        elif goal_scoring_rate > 1.5:
            shots_efficiency = 0.20
        elif goal_scoring_rate > 1.0:
            shots_efficiency = 0.15
        elif goal_scoring_rate > 0.5:
            shots_efficiency = 0.10
        else:
            shots_efficiency = 0.05
        
        shots_efficiency = shots_efficiency * (0.7 + clean_sheet_rate * 0.3)
        
        # Possession dominance
        goal_diff_per_match = goal_scoring_rate - goal_conceding_rate
        
        if goal_diff_per_match > 1.5:
            possession_dominance = 0.80
        elif goal_diff_per_match > 1.0:
            possession_dominance = 0.70
        elif goal_diff_per_match > 0.5:
            possession_dominance = 0.60
        elif goal_diff_per_match > 0:
            possession_dominance = 0.55
        elif goal_diff_per_match > -0.5:
            possession_dominance = 0.50
        elif goal_diff_per_match > -1.0:
            possession_dominance = 0.45
        else:
            possession_dominance = 0.40
        
        # Home/away advantage calculation
        home_matches = [m for m in matches if m.venue == 'H']
        away_matches = [m for m in matches if m.venue == 'A']
        
        home_advantage = 0.0
        away_performance = 0.5
        home_away_trend = 0.0
        
        if home_matches:
            home_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in home_matches)
            max_home_points = len(home_matches) * 3
            home_performance = home_points / max_home_points if max_home_points > 0 else 0
        
        if away_matches:
            away_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in away_matches)
            max_away_points = len(away_matches) * 3
            away_performance = away_points / max_away_points if max_away_points > 0 else 0
        
        if home_matches and away_matches:
            home_advantage = home_performance - away_performance
            home_away_trend = home_performance - away_performance
        elif home_matches:
            home_advantage = home_performance - 0.5
        elif away_matches:
            home_advantage = 0.5 - away_performance
        
        # Momentum calculation
        if len(matches) >= 6:
            last_3 = matches[:3]
            previous_3 = matches[3:6]
            
            last_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in last_3)
            previous_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in previous_3)
            
            max_points = 9
            momentum = (last_3_points - previous_3_points) / max_points
            
            recent_momentum = (last_3_points / 9 - 0.5) * 2
        else:
            momentum = 0.0
            recent_momentum = (recent_points / (recent_matches * 3) - 0.5) * 2 if recent_matches > 0 else 0.0
        
        # Goal trends
        if len(matches) >= 6:
            last_3_goals = sum(m.goals_scored for m in matches[:3])
            previous_3_goals = sum(m.goals_scored for m in matches[3:6])
            
            goal_trend = (last_3_goals - previous_3_goals) / 3.0
            goal_trend = max(-1.0, min(1.0, goal_trend / 2.0))
            
            last_3_conceded = sum(m.goals_conceded for m in matches[:3])
            previous_3_conceded = sum(m.goals_conceded for m in matches[3:6])
            
            defense_trend = (previous_3_conceded - last_3_conceded) / 3.0
            defense_trend = max(-1.0, min(1.0, defense_trend / 2.0))
        else:
            goal_trend = 0.0
            defense_trend = 0.0
        
        # FOOTBALL-SPECIFIC CONSISTENCY CALCULATION
        consistency_score = self.calculate_football_consistency(matches, match_results)
        
        # Simple result consistency for reference
        if match_results:
            unique_results = len(set(match_results))
            result_consistency = 1 - (unique_results / 3)
        else:
            result_consistency = 0.5
        
        # NEW METRICS: Based on wrong predictions analysis
        draws = sum(1 for m in matches if m.result == 'D')
        draw_tendency = draws / total_matches if total_matches > 0 else 0.3
        
        # Big win/loss tendency
        big_wins = sum(1 for m in matches if m.result == 'W' and m.goals_scored - m.goals_conceded >= 2)
        big_losses = sum(1 for m in matches if m.result == 'L' and m.goals_conceded - m.goals_scored >= 2)
        
        big_win_tendency = big_wins / total_matches if total_matches > 0 else 0.1
        big_loss_tendency = big_losses / total_matches if total_matches > 0 else 0.1
        
        # Comeback ability (winning after conceding first)
        wins = sum(1 for m in matches if m.result == 'W')
        comeback_ability = wins / max(1, total_matches - draws) * 0.7 if total_matches > draws else 0.5
        
        # Hold lead ability (not losing after scoring first)
        hold_lead_ability = 1 - big_loss_tendency
        
        # Performance variance
        if len(goals_scored) > 1:
            scoring_variance = statistics.variance(goals_scored) if len(goals_scored) > 1 else 0
            conceding_variance = statistics.variance(goals_conceded) if len(goals_conceded) > 1 else 0
            performance_variance = (scoring_variance + conceding_variance) / 2
        else:
            performance_variance = 0.0
        
        # Match type adjustment
        match_type_adjustment = 1.0
        if match_type == MatchType.CUP:
            match_type_adjustment = 0.95
        elif match_type == MatchType.FRIENDLY:
            match_type_adjustment = 0.8
        
        # Enhanced metrics
        recent_fixture_difficulty = 1.0 - (avg_opponent_position / 20) if avg_opponent_position else 0.5
        days_since_last_match = 7  # Default - would need actual match dates
        
        return TeamMetrics(
            team_name=team_name,
            form_rating=form_rating,
            recent_form_rating=recent_form_rating,
            goal_scoring_rate=goal_scoring_rate,
            goal_conceding_rate=goal_conceding_rate,
            clean_sheet_rate=clean_sheet_rate,
            top6_opponents_faced=top6_opponents,
            performance_vs_top6=performance_vs_top6,
            top6_performance_score=top6_performance_score,
            avg_opponent_position=avg_opponent_position,
            opponent_strength_factor=opponent_strength_factor,
            scoring_first_rate=scoring_first_rate,
            shots_efficiency=shots_efficiency,
            possession_dominance=possession_dominance,
            home_advantage=home_advantage,
            away_performance=away_performance,
            league_position=league_position,
            momentum=momentum,
            recent_momentum=recent_momentum,
            goal_trend=goal_trend,
            defense_trend=defense_trend,
            consistency_score=consistency_score,
            vs_bottom_half_performance=vs_bottom_half_performance,
            home_scoring_rate=home_scoring_rate,
            away_scoring_rate=away_scoring_rate,
            home_conceding_rate=home_conceding_rate,
            away_conceding_rate=away_conceding_rate,
            home_away_trend=home_away_trend,
            result_consistency=result_consistency,
            performance_variance=performance_variance,
            match_type_adjustment=match_type_adjustment,
            league_tier_factor=1.0,
            position_trend=0,
            h2h_metrics=h2h_metrics or H2HMetrics(),
            late_goal_tendency=0.3,
            comeback_ability=comeback_ability,
            hold_lead_ability=hold_lead_ability,
            draw_tendency=draw_tendency,
            big_win_tendency=big_win_tendency,
            big_loss_tendency=big_loss_tendency,
            recent_fixture_difficulty=recent_fixture_difficulty,
            days_since_last_match=days_since_last_match,
            injury_impact_score=1.0,
            manager_stability=1.0,
            motivation_factor=1.0
        )
    
    def analyze_core_factors(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics) -> CoreAnalysis:
        """Analyze core match determinants with enhanced factors"""
        
        analysis = CoreAnalysis()
        key_factors = []
        
        # 1. Form difference
        analysis.form_difference = home_metrics.form_rating - away_metrics.form_rating
        analysis.recent_form_difference = home_metrics.recent_form_rating - away_metrics.recent_form_rating
        
        if abs(analysis.form_difference) > 0.3:
            key_factors.append(f"STRONG FORM DIFFERENCE: {analysis.form_difference:+.2f}")
        elif abs(analysis.form_difference) > 0.15:
            key_factors.append(f"Form difference: {analysis.form_difference:+.2f}")
        
        if abs(analysis.recent_form_difference) > 0.4:
            key_factors.append(f"CRITICAL: Very strong recent form difference")
        elif abs(analysis.recent_form_difference) > 0.25:
            key_factors.append(f"IMPORTANT: Significant recent form difference")
        
        # 2. Goal scoring difference with home/away context
        analysis.goal_scoring_difference = home_metrics.home_scoring_rate - away_metrics.away_scoring_rate
        
        if analysis.goal_scoring_difference > 1.0:
            key_factors.append(f"MAJOR ATTACK ADVANTAGE: Home attack much stronger")
        elif analysis.goal_scoring_difference > 0.5:
            key_factors.append(f"Attack advantage for home team")
        elif analysis.goal_scoring_difference < -1.0:
            key_factors.append(f"MAJOR ATTACK ADVANTAGE: Away attack much stronger")
        elif analysis.goal_scoring_difference < -0.5:
            key_factors.append(f"Attack advantage for away team")
        
        # 3. Goal conceding difference with home/away context
        analysis.goal_conceding_difference = home_metrics.home_conceding_rate - away_metrics.away_conceding_rate
        
        if analysis.goal_conceding_difference > 0.8:
            key_factors.append(f"DEFENSIVE WEAKNESS: Home defense much weaker")
        elif analysis.goal_conceding_difference > 0.4:
            key_factors.append(f"Home defensive concerns")
        elif analysis.goal_conceding_difference < -0.8:
            key_factors.append(f"DEFENSIVE WEAKNESS: Away defense much weaker")
        elif analysis.goal_conceding_difference < -0.4:
            key_factors.append(f"Away defensive concerns")
        
        # 4. Opponent quality difference
        analysis.opponent_quality_difference = home_metrics.opponent_strength_factor - away_metrics.opponent_strength_factor
        
        if analysis.opponent_quality_difference > 0.3:
            key_factors.append(f"Home faced STRONGER opponents recently")
        elif analysis.opponent_quality_difference < -0.3:
            key_factors.append(f"Away faced STRONGER opponents recently")
        
        # 5. Home advantage
        analysis.home_advantage_strength = home_metrics.home_advantage
        
        if home_metrics.home_advantage > 0.5:
            key_factors.append(f"VERY STRONG HOME ADVANTAGE")
        elif home_metrics.home_advantage > 0.3:
            key_factors.append(f"Strong home advantage")
        elif home_metrics.home_advantage < -0.3:
            key_factors.append(f"POOR HOME FORM (advantage: {home_metrics.home_advantage:+.2f})")
        
        if away_metrics.away_performance > 0.7:
            key_factors.append(f"EXCELLENT AWAY FORM ({away_metrics.away_performance:.2f})")
        elif away_metrics.away_performance > 0.6:
            key_factors.append(f"Good away form")
        
        # 6. Momentum difference
        analysis.momentum_difference = home_metrics.recent_momentum - away_metrics.recent_momentum
        
        if abs(analysis.momentum_difference) > 0.6:
            key_factors.append(f"STRONG MOMENTUM SHIFT detected")
        elif abs(analysis.momentum_difference) > 0.4:
            key_factors.append(f"Significant momentum difference")
        
        # 7. CONSISTENCY DIFFERENCE
        analysis.consistency_difference = home_metrics.consistency_score - away_metrics.consistency_score
        
        if abs(analysis.consistency_difference) > 0.4:
            if analysis.consistency_difference > 0:
                key_factors.append(f"MAJOR CONSISTENCY ADVANTAGE: Home team much more consistent")
            else:
                key_factors.append(f"MAJOR CONSISTENCY ADVANTAGE: Away team much more consistent")
        elif abs(analysis.consistency_difference) > 0.2:
            if analysis.consistency_difference > 0:
                key_factors.append(f"Consistency advantage: Home team")
            else:
                key_factors.append(f"Consistency advantage: Away team")
        elif abs(analysis.consistency_difference) < 0.1:
            key_factors.append(f"Similar consistency levels")
        
        # 8. Top6 Performance difference
        analysis.top6_performance_difference = home_metrics.top6_performance_score - away_metrics.top6_performance_score
        
        if abs(analysis.top6_performance_difference) > 4.0:
            key_factors.append(f"HIGH-QUALITY OPPONENT EXPERTISE difference")
        elif abs(analysis.top6_performance_difference) > 2.0:
            key_factors.append(f"Quality opponent performance difference")
        
        # 9. H2H dominance
        if home_metrics.h2h_metrics.total_matches >= 3:
            analysis.h2h_dominance = home_metrics.h2h_metrics.home_win_rate - home_metrics.h2h_metrics.away_win_rate
            
            if abs(analysis.h2h_dominance) > 0.5:
                key_factors.append(f"VERY STRONG H2H DOMINANCE pattern")
            elif abs(analysis.h2h_dominance) > 0.3:
                key_factors.append(f"Strong H2H pattern")
            elif home_metrics.h2h_metrics.draw_rate > 0.6:
                key_factors.append(f"EXTREME H2H DRAW TENDENCY ({home_metrics.h2h_metrics.draw_rate:.0%})")
            elif home_metrics.h2h_metrics.draw_rate > 0.45:
                key_factors.append(f"H2H draw tendency ({home_metrics.h2h_metrics.draw_rate:.0%})")
        
        # 10. Tactical mismatch
        analysis.tactical_mismatch = 0.0
        
        if home_metrics.home_scoring_rate > 1.8 and away_metrics.away_conceding_rate > 1.8:
            analysis.tactical_mismatch = 0.4
            key_factors.append(f"TACTICAL MISMATCH: Strong home attack vs weak away defense")
        elif away_metrics.away_scoring_rate > 1.8 and home_metrics.home_conceding_rate > 1.8:
            analysis.tactical_mismatch = -0.4
            key_factors.append(f"TACTICAL MISMATCH: Strong away attack vs weak home defense")
        elif home_metrics.home_scoring_rate > 1.5 and away_metrics.away_conceding_rate > 1.5:
            analysis.tactical_mismatch = 0.2
            key_factors.append(f"Tactical advantage: Home attack vs away defense")
        elif away_metrics.away_scoring_rate > 1.5 and home_metrics.home_conceding_rate > 1.5:
            analysis.tactical_mismatch = -0.2
            key_factors.append(f"Tactical advantage: Away attack vs home defense")
        
        # 11. Comeback ability advantage
        analysis.comeback_advantage = home_metrics.comeback_ability - away_metrics.comeback_ability
        
        if abs(analysis.comeback_advantage) > 0.3:
            if analysis.comeback_advantage > 0:
                key_factors.append(f"COMEBACK SPECIALIST: Home team")
            else:
                key_factors.append(f"COMEBACK SPECIALIST: Away team")
        
        # 12. Expected goals advantage
        home_expected = home_metrics.home_scoring_rate - home_metrics.home_conceding_rate
        away_expected = away_metrics.away_scoring_rate - away_metrics.away_conceding_rate
        analysis.expected_goals_advantage = home_expected - away_expected
        
        if abs(analysis.expected_goals_advantage) > 1.0:
            key_factors.append(f"MAJOR EXPECTED GOALS ADVANTAGE")
        
        # 13. Draw tendency analysis
        if home_metrics.draw_tendency > 0.5 and away_metrics.draw_tendency > 0.5:
            key_factors.append(f"BOTH TEAMS ARE DRAW SPECIALISTS")
        elif home_metrics.draw_tendency > 0.5:
            key_factors.append(f"Home team is a draw specialist")
        elif away_metrics.draw_tendency > 0.5:
            key_factors.append(f"Away team is a draw specialist")
        
        # 14. Big win/loss tendencies
        if home_metrics.big_win_tendency > 0.3:
            key_factors.append(f"Home team has BIG WIN capability")
        if away_metrics.big_win_tendency > 0.3:
            key_factors.append(f"Away team has BIG WIN capability")
        if home_metrics.big_loss_tendency > 0.3:
            key_factors.append(f"Home team prone to BIG LOSSES")
        if away_metrics.big_loss_tendency > 0.3:
            key_factors.append(f"Away team prone to BIG LOSSES")
        
        # 15. Pattern recognition
        patterns, pattern_adjustments, pattern_messages = self.pattern_recognizer.detect_patterns(
            home_metrics, away_metrics
        )
        
        analysis.detected_patterns = patterns
        analysis.value_adjustments = pattern_adjustments
        
        for i, pattern in enumerate(patterns):
            key_factors.append(f"PATTERN DETECTED: {pattern_messages[i]}")
        
        analysis.key_factors = key_factors
        return analysis
    
    def analyze_match_summary(self, probabilities: Dict, goal_markets: Dict,
                             home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                             core_analysis: CoreAnalysis) -> MatchSummary:
        """Analyze match characteristics with enhanced logic"""
        
        summary = MatchSummary()
        
        # Determine if it's a close contest
        prob_spread = max(probabilities.values()) - min(probabilities.values())
        summary.is_close_contest = prob_spread < 15
        
        # Determine if it's a mismatch
        form_diff = abs(core_analysis.form_difference)
        goal_diff = abs(core_analysis.goal_scoring_difference)
        
        summary.is_mismatch = (form_diff > 0.4 and goal_diff > 1.0) or prob_spread > 40
        
        # Check for derby
        home_words = set(home_metrics.team_name.lower().split())
        away_words = set(away_metrics.team_name.lower().split())
        common_words = home_words.intersection(away_words)
        
        derby_indicators = {'city', 'united', 'rovers', 'town', 'athletic', 'fc', 'cf', 'sc', 'real', 'sporting'}
        if len(common_words) >= 2 or any(word in derby_indicators for word in common_words):
            summary.is_derby = True
        
        # Expected goals analysis
        summary.expected_goals_total = goal_markets['total_expected_goals']
        
        # More conservative thresholds for goal markets
        summary.high_scoring_potential = goal_markets['over_25'] > 70
        summary.low_scoring_potential = goal_markets['over_25'] < 30
        summary.bts_likely = goal_markets['bts'] > 70
        
        # H2H patterns
        if home_metrics.h2h_metrics.total_matches >= 3:
            summary.home_dominant_h2h = home_metrics.h2h_metrics.home_win_rate > 0.65
            summary.away_dominant_h2h = home_metrics.h2h_metrics.away_win_rate > 0.65
            summary.draw_tendency_h2h = home_metrics.h2h_metrics.draw_rate > 0.5
        
        # Tactical advantage analysis
        if core_analysis.tactical_mismatch > 0.3:
            summary.tactical_advantage = f"STRONG: {home_metrics.team_name[:12]} attack vs weak {away_metrics.team_name[:12]} defense"
        elif core_analysis.tactical_mismatch > 0.1:
            summary.tactical_advantage = f"{home_metrics.team_name[:12]} attack advantage"
        elif core_analysis.tactical_mismatch < -0.3:
            summary.tactical_advantage = f"STRONG: {away_metrics.team_name[:12]} attack vs weak {home_metrics.team_name[:12]} defense"
        elif core_analysis.tactical_mismatch < -0.1:
            summary.tactical_advantage = f"{away_metrics.team_name[:12]} attack advantage"
        
        # New characteristics based on wrong predictions analysis
        summary.is_unpredictable = (
            abs(core_analysis.form_difference) < 0.2 and
            abs(core_analysis.recent_form_difference) < 0.3 and
            home_metrics.consistency_score < 0.6 and
            away_metrics.consistency_score < 0.6
        )
        
        summary.has_momentum_shift = abs(core_analysis.momentum_difference) > 0.5
        
        summary.is_defensive_battle = (
            home_metrics.goal_scoring_rate < 1.0 and
            away_metrics.goal_scoring_rate < 1.0 and
            home_metrics.clean_sheet_rate > 0.5 and
            away_metrics.clean_sheet_rate > 0.5
        )
        
        summary.is_goalfest_potential = (
            home_metrics.goal_scoring_rate > 2.0 or
            away_metrics.goal_scoring_rate > 2.0 or
            home_metrics.goal_conceding_rate > 2.0 or
            away_metrics.goal_conceding_rate > 2.0
        )
        
        # Goal scoring advantages
        summary.home_goal_scoring_advantage = home_metrics.home_scoring_rate - away_metrics.away_conceding_rate
        summary.away_goal_scoring_advantage = away_metrics.away_scoring_rate - home_metrics.home_conceding_rate
        summary.home_defensive_advantage = away_metrics.away_scoring_rate - home_metrics.home_conceding_rate
        summary.away_defensive_advantage = home_metrics.home_scoring_rate - away_metrics.away_conceding_rate
        
        # Build characteristics list
        characteristics = []
        
        if summary.is_close_contest:
            characteristics.append("CLOSE CONTEST - Small probability spread")
        elif summary.is_mismatch:
            characteristics.append("CLEAR FAVORITE - Significant mismatch")
        
        if summary.is_derby:
            characteristics.append("LOCAL DERBY - Highly unpredictable")
        
        if summary.is_unpredictable:
            characteristics.append("UNPREDICTABLE - Many variables")
        
        if summary.has_momentum_shift:
            characteristics.append("MOMENTUM SHIFT - Recent form change")
        
        if summary.is_defensive_battle:
            characteristics.append("DEFENSIVE BATTLE - Low scoring expected")
        elif summary.is_goalfest_potential:
            characteristics.append("GOALFEST POTENTIAL - High scoring possible")
        
        if abs(core_analysis.recent_form_difference) > 0.3:
            if core_analysis.recent_form_difference > 0:
                characteristics.append(f"RECENT FORM: {home_metrics.team_name[:12]} superior")
            else:
                characteristics.append(f"RECENT FORM: {away_metrics.team_name[:12]} superior")
        
        if home_metrics.home_advantage > 0.4:
            characteristics.append(f"HOME ADVANTAGE: {home_metrics.team_name[:12]} very strong at home")
        elif away_metrics.away_performance > 0.7:
            characteristics.append(f"AWAY FORM: {away_metrics.team_name[:12]} excellent away")
        
        if abs(core_analysis.top6_performance_difference) > 3.0:
            if core_analysis.top6_performance_difference > 0:
                characteristics.append(f"QUALITY: {home_metrics.team_name[:12]} excels vs top teams")
            else:
                characteristics.append(f"QUALITY: {away_metrics.team_name[:12]} excels vs top teams")
        
        if summary.high_scoring_potential:
            characteristics.append("HIGH SCORING: >2.5 goals likely")
        elif summary.low_scoring_potential:
            characteristics.append("LOW SCORING: <2.5 goals likely")
        
        if summary.bts_likely:
            characteristics.append("BOTH TEAMS TO SCORE: Likely")
        
        if summary.home_dominant_h2h:
            characteristics.append(f"H2H DOMINANCE: {home_metrics.team_name[:12]}")
        elif summary.away_dominant_h2h:
            characteristics.append(f"H2H DOMINANCE: {away_metrics.team_name[:12]}")
        elif summary.draw_tendency_h2h:
            characteristics.append("H2H PATTERN: Strong draw tendency")
        
        if summary.tactical_advantage:
            characteristics.append(f"TACTICAL: {summary.tactical_advantage}")
        
        # Add detected patterns
        if core_analysis.detected_patterns:
            for i, pattern in enumerate(core_analysis.detected_patterns[:2]):
                characteristics.append(f"PATTERN: {pattern}")
        
        # Add key factors from core analysis
        characteristics.extend(core_analysis.key_factors[:4])
        
        summary.characteristics = characteristics
        
        return summary
    
    def calculate_probabilities_with_learning(self, home_metrics: TeamMetrics, 
                                            away_metrics: TeamMetrics,
                                            core_analysis: CoreAnalysis,
                                            match_summary: MatchSummary,
                                            factors: MatchFactors,
                                            league: str = "Unknown"):
        """Calculate match probabilities with learning adjustments including league-specific factors"""
        
        # Get league-specific adjustments
        league_adj = self.get_league_adjustments(league)
        
        home_score = 0
        away_score = 0
        
        # 1. Form rating (15%)
        home_score += home_metrics.form_rating * factors.form_weight
        away_score += away_metrics.form_rating * factors.form_weight
        
        # 2. Recent form (14%)
        home_score += home_metrics.recent_form_rating * factors.recent_form_weight
        away_score += away_metrics.recent_form_rating * factors.recent_form_weight
        
        # 3. Attack vs Defense (20%)
        home_attack_vs_defense = (home_metrics.home_scoring_rate * 0.6 + 
                                 (2 - away_metrics.away_conceding_rate) * 0.4) / 2
        away_attack_vs_defense = (away_metrics.away_scoring_rate * 0.6 + 
                                 (2 - home_metrics.home_conceding_rate) * 0.4) / 2
        
        home_score += home_attack_vs_defense * factors.attack_vs_defense_weight
        away_score += away_attack_vs_defense * factors.attack_vs_defense_weight
        
        # 4. Opponent quality (18%)
        home_opponent_quality = home_metrics.opponent_strength_factor * 0.6 + home_metrics.performance_vs_top6 * 0.4
        away_opponent_quality = away_metrics.opponent_strength_factor * 0.6 + away_metrics.performance_vs_top6 * 0.4
        
        home_score += home_opponent_quality * factors.opponent_quality_weight
        away_score += away_opponent_quality * factors.opponent_quality_weight
        
        # 5. Home advantage (12%) - ENHANCED WITH LEAGUE-SPECIFIC ADJUSTMENTS
        home_advantage_factor = (home_metrics.home_advantage + 1) / 2
        away_disadvantage_factor = (1 - home_metrics.home_advantage) / 2
        
        # Apply league-specific home advantage multiplier
        home_advantage_factor *= league_adj['home_advantage_multiplier']
        
        # If away team is strong away, reduce home advantage
        if away_metrics.away_performance > 0.7:
            home_advantage_factor *= 0.8
            away_disadvantage_factor *= 1.2
        
        home_score += home_advantage_factor * factors.home_advantage_weight
        away_score += away_disadvantage_factor * factors.home_advantage_weight
        
        # 6. Momentum (6%)
        home_momentum = (home_metrics.recent_momentum + 1) / 2
        away_momentum = (away_metrics.recent_momentum + 1) / 2
        
        home_score += home_momentum * factors.momentum_weight
        away_score += away_momentum * factors.momentum_weight
        
        # 7. Consistency (10%)
        home_score += home_metrics.consistency_score * factors.consistency_weight
        away_score += away_metrics.consistency_score * factors.consistency_weight
        
        # 8. Top6 Performance (8%)
        home_score += (home_metrics.top6_performance_score / 10) * factors.top6_performance_weight
        away_score += (away_metrics.top6_performance_score / 10) * factors.top6_performance_weight
        
        # 9. H2H factor (16%)
        if home_metrics.h2h_metrics.total_matches >= 3:
            h2h_weight = min(0.2, 0.12 + (home_metrics.h2h_metrics.total_matches / 50))
            
            if home_metrics.h2h_metrics.home_win_rate > home_metrics.h2h_metrics.away_win_rate + 0.35:
                h2h_home_advantage = 0.3
            elif home_metrics.h2h_metrics.home_win_rate > home_metrics.h2h_metrics.away_win_rate + 0.2:
                h2h_home_advantage = 0.15
            elif home_metrics.h2h_metrics.away_win_rate > home_metrics.h2h_metrics.home_win_rate + 0.35:
                h2h_home_advantage = -0.3
            elif home_metrics.h2h_metrics.away_win_rate > home_metrics.h2h_metrics.home_win_rate + 0.2:
                h2h_home_advantage = -0.15
            elif home_metrics.h2h_metrics.draw_rate > 0.55:
                h2h_home_advantage = 0
                home_score *= 0.9
                away_score *= 0.9
            else:
                h2h_home_advantage = 0
            
            home_score += (0.5 + h2h_home_advantage) * h2h_weight
            away_score += (0.5 - h2h_home_advantage) * h2h_weight
        
        # 10. Tactical mismatch (5%)
        tactical_factor = (home_metrics.home_scoring_rate - away_metrics.away_conceding_rate) * 0.1
        home_score += max(0, tactical_factor) * factors.tactical_mismatch_weight
        away_score += max(0, -tactical_factor) * factors.tactical_mismatch_weight
        
        # 11. Comeback ability (3%)
        home_score += home_metrics.comeback_ability * factors.comeback_ability_weight
        away_score += away_metrics.comeback_ability * factors.comeback_ability_weight
        
        # 12. Draw tendency (5%)
        draw_tendency_factor = (home_metrics.draw_tendency + away_metrics.draw_tendency) / 2
        # This factor increases draw probability later
        
        # 13. Fixture congestion (4%)
        # Simplified - reduce score for team with less rest
        if home_metrics.days_since_last_match < away_metrics.days_since_last_match:
            home_score *= 0.95
        elif home_metrics.days_since_last_match > away_metrics.days_since_last_match:
            away_score *= 0.95
        
        # Apply match type adjustments
        home_score *= home_metrics.match_type_adjustment
        away_score *= away_metrics.match_type_adjustment
        
        # Apply pattern adjustments
        if core_analysis.value_adjustments:
            pattern_adjustment = sum(core_analysis.value_adjustments)
            if pattern_adjustment > 0:
                home_score *= (1 + pattern_adjustment)
            elif pattern_adjustment < 0:
                away_score *= (1 - pattern_adjustment)
        
        # Base draw probability with league-specific adjustments
        closeness = 1 - abs(home_score - away_score)
        base_draw = league_adj['draw_rate_expected'] * closeness
        
        # Adjust for draw tendencies
        if draw_tendency_factor > 0.4:
            base_draw *= 1.3
        
        # Conference League specific: High draw rate, very low scoring
        if league == "Conference League":
            base_draw *= 1.8  # High draw boost (60% draw rate observed)
            
        # Europa League specific: EXTREME draw rate, almost no decisive results
        elif league == "Europa League":
            base_draw *= 2.5  # EXTREME draw boost (89% draw rate observed!)
            
        # Champions League specific: Higher scoring, less draws
        elif league == "Champions League":
            base_draw *= 0.85  # Reduce draw probability
            # Boost goal expectations
            
        # Premier League specific: Lower draw rate, more decisive results
        elif league == "Premier League":
            base_draw *= 0.9  # Reduce draw probability slightly
        
        # Adjust for low scoring teams
        if home_metrics.goal_scoring_rate < 1.0 and away_metrics.goal_scoring_rate < 1.0:
            base_draw *= 1.2
        
        # Normalize
        total = home_score + away_score + base_draw
        home_prob = home_score / total * 100
        away_prob = away_score / total * 100
        draw_prob = base_draw / total * 100
        
        return {
            'home_win': home_prob,
            'away_win': away_prob,
            'draw': draw_prob,
            'home_score': home_score,
            'away_score': away_score,
            'draw_score': base_draw
        }
    
    def make_decisions(self, probabilities: Dict, goal_markets: Dict,
                      home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                      core_analysis: CoreAnalysis, match_summary: MatchSummary,
                      factors: MatchFactors) -> Dict:
        """Make betting decisions with enhanced logic and risk management"""
        
        decisions = {
            'primary_bet': None,
            'secondary_bets': [],
            'alternative_bets': [],
            'confidence': 'LOW',
            'confidence_score': 0,
            'reasoning': [],
            'avoid_bets': [],
            'key_insights': [],
            'match_characteristics': match_summary.characteristics,
            'risk_level': 'HIGH',
            'stake_recommendation': 'VERY SMALL',
            'value_bets': [],
            'learning_insights': []
        }
        
        # CONFIDENCE SCORE CALCULATION (0-15 scale)
        confidence_score = 0
        
        # 1. Very strong form difference (+4)
        if abs(core_analysis.form_difference) > 0.4:
            confidence_score += 4
            decisions['key_insights'].append(f"VERY strong form difference")
        elif abs(core_analysis.form_difference) > 0.25:
            confidence_score += 2
        
        # 2. Extremely strong recent form difference (+5)
        if abs(core_analysis.recent_form_difference) > 0.5:
            confidence_score += 5
            decisions['key_insights'].append(f"EXTREMELY strong recent form difference")
        elif abs(core_analysis.recent_form_difference) > 0.35:
            confidence_score += 3
        elif abs(core_analysis.recent_form_difference) > 0.2:
            confidence_score += 2
        
        # 3. Major goal difference (+4)
        if abs(core_analysis.goal_scoring_difference) > 1.2 or abs(core_analysis.goal_conceding_difference) > 1.2:
            confidence_score += 4
            decisions['key_insights'].append(f"Major goal difference")
        elif abs(core_analysis.goal_scoring_difference) > 0.8 or abs(core_analysis.goal_conceding_difference) > 0.8:
            confidence_score += 2
        
        # 4. Very strong home advantage (+3)
        if home_metrics.home_advantage > 0.5:
            confidence_score += 3
            decisions['key_insights'].append(f"VERY strong home advantage")
        elif home_metrics.home_advantage > 0.3:
            confidence_score += 2
        elif home_metrics.home_advantage > 0.2:
            confidence_score += 1
        
        # 5. Excellent away performance (+3)
        if away_metrics.away_performance > 0.75:
            confidence_score += 3
            decisions['key_insights'].append(f"Excellent away performance")
        elif away_metrics.away_performance > 0.65:
            confidence_score += 2
        
        # 6. Performance against top teams (+3)
        if abs(core_analysis.top6_performance_difference) > 4.0:
            confidence_score += 3
        elif abs(core_analysis.top6_performance_difference) > 2.5:
            confidence_score += 2
        
        # 7. Clear momentum (+2)
        if abs(core_analysis.momentum_difference) > 0.5:
            confidence_score += 2
        
        # 8. Strong H2H pattern (+5 if very strong, +3 if strong)
        if home_metrics.h2h_metrics.total_matches >= 5:
            if abs(core_analysis.h2h_dominance) > 0.6:
                confidence_score += 5
                decisions['key_insights'].append("VERY STRONG historical dominance")
            elif abs(core_analysis.h2h_dominance) > 0.4:
                confidence_score += 3
                decisions['key_insights'].append("Strong historical pattern")
            elif home_metrics.h2h_metrics.draw_rate > 0.65:
                confidence_score += 4
                decisions['key_insights'].append("Extreme H2H draw pattern")
            elif home_metrics.h2h_metrics.draw_rate > 0.5:
                confidence_score += 3
                decisions['key_insights'].append("Strong H2H draw pattern")
        
        # 9. Consistency advantage (+2)
        if abs(core_analysis.consistency_difference) > 0.4:
            confidence_score += 2
            if core_analysis.consistency_difference > 0:
                decisions['key_insights'].append(f"Consistency advantage for home team")
            else:
                decisions['key_insights'].append(f"Consistency advantage for away team")
        
        # 10. Strong tactical mismatch (+3)
        if abs(core_analysis.tactical_mismatch) > 0.3:
            confidence_score += 3
            if core_analysis.tactical_mismatch > 0:
                decisions['key_insights'].append(f"STRONG tactical advantage for home")
            else:
                decisions['key_insights'].append(f"STRONG tactical advantage for away")
        elif abs(core_analysis.tactical_mismatch) > 0.2:
            confidence_score += 2
        
        # CONSOLIDATE CONFIDENCE: Apply consistency adjustments
        inconsistency_penalty = 0
        
        if home_metrics.consistency_score < 0.3:
            inconsistency_penalty += 3
            decisions['key_insights'].append(f"Home team has LOW consistency ({home_metrics.consistency_score:.2f})")
        elif home_metrics.consistency_score < 0.5:
            inconsistency_penalty += 1
        
        if away_metrics.consistency_score < 0.3:
            inconsistency_penalty += 3
            decisions['key_insights'].append(f"Away team has LOW consistency ({away_metrics.consistency_score:.2f})")
        elif away_metrics.consistency_score < 0.5:
            inconsistency_penalty += 1
        
        # Apply inconsistency penalty
        confidence_score = max(0, confidence_score - inconsistency_penalty)
        
        # Set confidence level based on ADJUSTED score
        decisions['confidence_score'] = confidence_score
        
        # Set confidence, risk, and stake based on adjusted score
        if confidence_score >= 12:
            decisions['confidence'] = 'VERY HIGH'
            decisions['risk_level'] = 'LOW'
            decisions['stake_recommendation'] = 'MEDIUM-HIGH'
        elif confidence_score >= 9:
            decisions['confidence'] = 'HIGH'
            decisions['risk_level'] = 'LOW-MEDIUM'
            decisions['stake_recommendation'] = 'MEDIUM'
        elif confidence_score >= 6:
            decisions['confidence'] = 'MEDIUM'
            decisions['risk_level'] = 'MEDIUM'
            decisions['stake_recommendation'] = 'SMALL-MEDIUM'
        elif confidence_score >= 3:
            decisions['confidence'] = 'LOW'
            decisions['risk_level'] = 'HIGH'
            decisions['stake_recommendation'] = 'VERY SMALL'
        else:
            decisions['confidence'] = 'VERY LOW'
            decisions['risk_level'] = 'VERY HIGH'
            decisions['stake_recommendation'] = 'AVOID'
        
        # SPECIAL CASE: If confidence is high but consistency is low, downgrade
        if decisions['confidence'] in ['HIGH', 'VERY HIGH'] and (home_metrics.consistency_score < 0.4 or away_metrics.consistency_score < 0.4):
            decisions['confidence'] = 'MEDIUM'
            decisions['risk_level'] = 'MEDIUM-HIGH'
            decisions['stake_recommendation'] = 'SMALL'
            decisions['key_insights'].append(f"Confidence downgraded due to inconsistency")
        
        # Add learning insights
        if self.learning_engine.learning_metrics.total_predictions > 0:
            recent_acc = self.learning_engine.calculate_recent_accuracy(20)
            if recent_acc > 0.65:
                decisions['learning_insights'].append(f"Model performing well recently ({recent_acc:.0%} accuracy)")
            elif recent_acc < 0.40:
                decisions['learning_insights'].append(f"Model struggling recently ({recent_acc:.0%} accuracy)")
        
        # Apply final probability adjustments based on learning
        home_prob = probabilities['home_win']
        away_prob = probabilities['away_win']
        draw_prob = probabilities['draw']
        
        # Derbies are more unpredictable
        if match_summary.is_derby:
            home_prob = home_prob * 0.85
            away_prob = away_prob * 0.85
            draw_prob = min(45, draw_prob * 1.3)
            decisions['reasoning'].append("Derby match - increased uncertainty")
        
        # Unpredictable matches
        if match_summary.is_unpredictable:
            home_prob = home_prob * 0.9
            away_prob = away_prob * 0.9
            draw_prob = min(40, draw_prob * 1.2)
            decisions['reasoning'].append("Unpredictable match - caution advised")
        
        # Normalize probabilities
        total = home_prob + away_prob + draw_prob
        if total > 0:
            home_prob = home_prob / total * 100
            away_prob = away_prob / total * 100
            draw_prob = draw_prob / total * 100
        
        # VALUE BET DETECTION
        # Mock market odds for demonstration - in real use, fetch actual odds
        mock_market_odds = {'1': 2.0, 'X': 3.3, '2': 3.5}
        value_bets = self.value_detector.find_value_bets({
            'home_win': home_prob,
            'draw': draw_prob,
            'away_win': away_prob
        }, mock_market_odds)
        
        decisions['value_bets'] = value_bets
        
        # Make primary bet decision
        primary_reasoning = []
        
        # EMERGENCY DRAW DETECTION - Critical fix for draw prediction blindness
        prob_diff = abs(home_prob - away_prob)
        h2h_draw_rate = (home_metrics.h2h_metrics.draws / max(home_metrics.h2h_metrics.total_matches, 1)) * 100 if home_metrics.h2h_metrics.total_matches > 0 else 0
        
        # Draw detection conditions - PRIORITY OVER VALUE BETS
        if (prob_diff < 7.0 and h2h_draw_rate > 25.0 and home_metrics.h2h_metrics.total_matches >= 5):
            decisions['primary_bet'] = {
                'type': 'MATCH_RESULT',
                'selection': 'DRAW',
                'probability': max(draw_prob, 35.0),
                'odds_range': '3.00-3.50',
                'stake_suggestion': decisions['stake_recommendation'],
                'reasoning': 'Emergency draw detection: Close probabilities + high H2H draw rate'
            }
            primary_reasoning.append(f"Close probabilities ({prob_diff:.1f}% difference)")
            primary_reasoning.append(f"H2H draw rate: {h2h_draw_rate:.1f}% ({home_metrics.h2h_metrics.draws}/{home_metrics.h2h_metrics.total_matches})")
        
        # Check for strong value bets second
        elif value_bets and value_bets[0]['value_percentage'] > 10:
            best_value = value_bets[0]
            selection_map = {
                'home_win': f"{home_metrics.team_name[:12]} Win",
                'away_win': f"{away_metrics.team_name[:12]} Win",
                'draw': 'Draw'
            }
            
            decisions['primary_bet'] = {
                'type': 'MATCH_RESULT',
                'selection': best_value['outcome'].upper(),
                'probability': best_value['predicted_prob'],
                'odds_range': f"{1/best_value['predicted_prob']*100:.2f}",
                'stake_suggestion': decisions['stake_recommendation'],
                'reasoning': f"High value bet ({best_value['value_percentage']:.1f}% value)",
                'value_score': best_value['value_percentage']
            }
            primary_reasoning.append(f"Value bet: {best_value['value_percentage']:.1f}% edge")
        
        # EMERGENCY DRAW DETECTION - Critical fix for draw prediction blindness
        elif not decisions.get('primary_bet'):
            # Emergency draw detection algorithm
            prob_diff = abs(home_prob - away_prob)
            h2h_draw_rate = (home_metrics.h2h_metrics.draws / max(home_metrics.h2h_metrics.total_matches, 1)) * 100 if home_metrics.h2h_metrics.total_matches > 0 else 0
            
            # Draw detection conditions
            if (prob_diff < 7.0 and h2h_draw_rate > 25.0 and home_metrics.h2h_metrics.total_matches >= 5):
                decisions['primary_bet'] = {
                    'type': 'MATCH_RESULT',
                    'selection': 'DRAW',
                    'probability': max(draw_prob, 35.0),
                    'odds_range': '3.00-3.50',
                    'stake_suggestion': decisions['stake_recommendation'],
                    'reasoning': 'Emergency draw detection: Close probabilities + high H2H draw rate'
                }
                primary_reasoning.append(f"Close probabilities ({prob_diff:.1f}% difference)")
                primary_reasoning.append(f"H2H draw rate: {h2h_draw_rate:.1f}% ({home_metrics.h2h_metrics.draws}/{home_metrics.h2h_metrics.total_matches})")
                
            # For mismatches, require higher thresholds - ADJUSTED FOR HIGH CONFIDENCE
            if match_summary.is_mismatch:
                if (home_prob > away_prob + 20 and home_prob > 60) or \
                   (decisions['confidence_score'] >= 12 and home_prob > away_prob + 8 and home_prob > 45):
                    decisions['primary_bet'] = {
                        'type': 'MATCH_RESULT',
                        'selection': 'HOME_WIN',
                        'probability': home_prob,
                        'odds_range': '1.40-1.80' if home_prob > 70 else '1.80-2.20',
                        'stake_suggestion': decisions['stake_recommendation'],
                        'reasoning': 'Clear favorite with multiple advantages'
                    }
                    primary_reasoning.append("Clear favorite based on multiple factors")
                    
                elif (away_prob > home_prob + 20 and away_prob > 60) or \
                     (decisions['confidence_score'] >= 12 and away_prob > home_prob + 8 and away_prob > 45):
                    decisions['primary_bet'] = {
                        'type': 'MATCH_RESULT',
                        'selection': 'AWAY_WIN',
                        'probability': away_prob,
                        'odds_range': '1.90-2.40' if away_prob > 65 else '2.40-3.00',
                        'stake_suggestion': decisions['stake_recommendation'],
                        'reasoning': 'Clear favorite with multiple advantages'
                    }
                    primary_reasoning.append("Clear favorite based on multiple factors")
            else:
                # Regular match decision
                draw_indicators = [
                    match_summary.draw_tendency_h2h,
                    abs(core_analysis.form_difference) < 0.15,
                    home_metrics.draw_tendency > 0.4 or away_metrics.draw_tendency > 0.4,
                    match_summary.is_close_contest and draw_prob > 35
                ]
                
                strong_draw_indicators = sum(draw_indicators)
                
                if strong_draw_indicators >= 2 and draw_prob > 35:
                    decisions['primary_bet'] = {
                        'type': 'MATCH_RESULT',
                        'selection': 'DRAW',
                        'probability': draw_prob,
                        'odds_range': '3.00-3.50',
                        'stake_suggestion': decisions['stake_recommendation'],
                        'reasoning': 'Multiple strong draw indicators'
                    }
                    
                    primary_reasoning = []
                    if match_summary.draw_tendency_h2h:
                        primary_reasoning.append(f"H2H: {home_metrics.h2h_metrics.draws}/{home_metrics.h2h_metrics.total_matches} draws")
                    if match_summary.is_close_contest:
                        primary_reasoning.append("Close contest")
                    if home_metrics.draw_tendency > 0.4:
                        primary_reasoning.append(f"Home draw tendency: {home_metrics.draw_tendency:.0%}")
                    if away_metrics.draw_tendency > 0.4:
                        primary_reasoning.append(f"Away draw tendency: {away_metrics.draw_tendency:.0%}")
                
                # Home win decision - ADJUSTED FOR HIGH CONFIDENCE
                elif (home_prob > away_prob + 10 and home_prob > 45) or \
                     (decisions['confidence_score'] >= 12 and home_prob > away_prob + 5 and home_prob > 40):
                    decisions['primary_bet'] = {
                        'type': 'MATCH_RESULT',
                        'selection': 'HOME_WIN',
                        'probability': home_prob,
                        'odds_range': '1.80-2.30' if home_prob > 55 else '2.30-2.80',
                        'stake_suggestion': decisions['stake_recommendation'],
                        'reasoning': 'Home advantage and form'
                    }
                    
                    if core_analysis.recent_form_difference > 0.25:
                        primary_reasoning.append(f"Better recent form ({home_metrics.recent_form_rating:.2f} vs {away_metrics.recent_form_rating:.2f})")
                    if home_metrics.home_advantage > 0.3:
                        primary_reasoning.append(f"Strong home advantage")
                    if match_summary.home_dominant_h2h:
                        primary_reasoning.append(f"H2H dominance")
                    if core_analysis.top6_performance_difference > 2.0:
                        primary_reasoning.append(f"Stronger vs top teams")
                    if match_summary.home_goal_scoring_advantage > 0.8:
                        primary_reasoning.append(f"Significant attacking advantage")
                    if core_analysis.consistency_difference > 0.2:
                        primary_reasoning.append(f"More consistent than opponent")
                
                # Away win decision - ADJUSTED FOR HIGH CONFIDENCE  
                elif (away_prob > home_prob + 10 and away_prob > 45) or \
                     (decisions['confidence_score'] >= 12 and away_prob > home_prob + 5 and away_prob > 40):
                    decisions['primary_bet'] = {
                        'type': 'MATCH_RESULT',
                        'selection': 'AWAY_WIN',
                        'probability': away_prob,
                        'odds_range': '2.20-2.80' if away_prob > 50 else '2.80-3.50',
                        'stake_suggestion': decisions['stake_recommendation'],
                        'reasoning': 'Away form and performance'
                    }
                    
                    if core_analysis.recent_form_difference < -0.25:
                        primary_reasoning.append(f"Better recent form ({away_metrics.recent_form_rating:.2f} vs {home_metrics.recent_form_rating:.2f})")
                    if match_summary.away_dominant_h2h:
                        primary_reasoning.append(f"H2H dominance")
                    if core_analysis.top6_performance_difference < -2.0:
                        primary_reasoning.append(f"Stronger vs top teams")
                    if away_metrics.away_performance > 0.7:
                        primary_reasoning.append(f"Excellent away form")
                    if match_summary.away_goal_scoring_advantage > 0.8:
                        primary_reasoning.append(f"Significant attacking advantage")
                    if core_analysis.consistency_difference < -0.2:
                        primary_reasoning.append(f"More consistent than opponent")
        
        # Add detailed reasoning if available
        if primary_reasoning and decisions.get('primary_bet'):
            decisions['primary_bet']['detailed_reasoning'] = " | ".join(primary_reasoning)
        
        # GOAL MARKET BETS
        if goal_markets['over_25'] > 75:
            decisions['secondary_bets'].append({
                'type': 'GOALS',
                'selection': 'OVER_2.5',
                'probability': goal_markets['over_25'],
                'confidence': 'HIGH' if goal_markets['over_25'] > 85 else 'MEDIUM',
                'reasoning': f"High scoring teams with weak defenses"
            })
        elif goal_markets['over_25'] < 25:
            decisions['secondary_bets'].append({
                'type': 'GOALS',
                'selection': 'UNDER_2.5',
                'probability': 100 - goal_markets['over_25'],
                'confidence': 'HIGH' if goal_markets['over_25'] < 15 else 'MEDIUM',
                'reasoning': f"Strong defenses and/or poor attacks"
            })
        
        # Both Teams to Score
        if goal_markets['bts'] > 75:
            decisions['secondary_bets'].append({
                'type': 'BOTH_TEAMS_SCORE',
                'selection': 'YES',
                'probability': goal_markets['bts'],
                'confidence': 'HIGH' if goal_markets['bts'] > 85 else 'MEDIUM',
                'reasoning': f"Both teams consistently score"
            })
        elif goal_markets['bts'] < 25:
            decisions['secondary_bets'].append({
                'type': 'BOTH_TEAMS_SCORE',
                'selection': 'NO',
                'probability': 100 - goal_markets['bts'],
                'confidence': 'HIGH' if goal_markets['bts'] < 15 else 'MEDIUM',
                'reasoning': f"Strong defensive records"
            })
        
        # RISK WARNINGS
        if home_metrics.recent_form_rating < 0.25:
            decisions['avoid_bets'].append(f"AVOID {home_metrics.team_name[:10]} - VERY POOR recent form")
        elif home_metrics.form_rating < 0.3:
            decisions['avoid_bets'].append(f"Caution: {home_metrics.team_name[:10]} - poor form")
        
        if away_metrics.recent_form_rating < 0.25:
            decisions['avoid_bets'].append(f"AVOID {away_metrics.team_name[:10]} - VERY POOR recent form")
        elif away_metrics.form_rating < 0.3:
            decisions['avoid_bets'].append(f"Caution: {away_metrics.team_name[:10]} - poor form")
        
        # Opponent strength warnings
        if home_metrics.avg_opponent_position > 16 and home_metrics.form_rating > 0.7:
            decisions['avoid_bets'].append(f"WARNING: {home_metrics.team_name[:10]} faced VERY WEAK opponents")
        if away_metrics.avg_opponent_position > 16 and away_metrics.form_rating > 0.7:
            decisions['avoid_bets'].append(f"WARNING: {away_metrics.team_name[:10]} faced VERY WEAK opponents")
        
        # Inconsistency warnings
        if home_metrics.consistency_score < 0.3:
            decisions['avoid_bets'].append(f"WARNING: {home_metrics.team_name[:10]} shows LOW consistency ({home_metrics.consistency_score:.2f})")
        elif home_metrics.consistency_score < 0.5:
            decisions['avoid_bets'].append(f"Note: {home_metrics.team_name[:10]} has moderate consistency")
        
        if away_metrics.consistency_score < 0.3:
            decisions['avoid_bets'].append(f"WARNING: {away_metrics.team_name[:10]} shows LOW consistency ({away_metrics.consistency_score:.2f})")
        elif away_metrics.consistency_score < 0.5:
            decisions['avoid_bets'].append(f"Note: {away_metrics.team_name[:10]} has moderate consistency")
        
        # High variance warning
        if home_metrics.performance_variance > 2.0 or away_metrics.performance_variance > 2.0:
            decisions['avoid_bets'].append("HIGH PERFORMANCE VARIANCE - Goal scoring unpredictable")
        
        # Derby warning
        if match_summary.is_derby:
            decisions['avoid_bets'].append("DERBY MATCH - Highly unpredictable, reduced stakes recommended")
        
        # Unpredictable match warning
        if match_summary.is_unpredictable:
            decisions['avoid_bets'].append("UNPREDICTABLE MATCH - Many conflicting indicators")
        
        # Add value bets to reasoning
        if decisions['value_bets']:
            for value_bet in decisions['value_bets'][:2]:
                decisions['reasoning'].append(f"VALUE BET: {value_bet['outcome']} ({value_bet['value_percentage']:.1f}% value)")
        
        # Add learning insights to reasoning
        if decisions['learning_insights']:
            decisions['reasoning'].extend(decisions['learning_insights'])
        
        return decisions
    
    def record_match_outcome(self, match_id: str, home_team: str, away_team: str,
                            predicted_result: str, predicted_probabilities: Dict,
                            confidence_score: int, key_factors: List[str],
                            match_characteristics: List[str],
                            home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                            core_analysis: CoreAnalysis, actual_result: str,
                            league: str = "Unknown"):
        """Record the outcome of a match for learning"""
        
        # Calculate error magnitude
        predicted_prob = predicted_probabilities.get({
            'H': 'home_win', 'A': 'away_win', 'D': 'draw'
        }[predicted_result], 50)
        
        if predicted_result == actual_result:
            error_magnitude = 0
        else:
            error_magnitude = predicted_prob / 100  # Higher confidence wrong = bigger error
        
        outcome = PredictionOutcome(
            match_id=match_id,
            timestamp=datetime.now(),
            home_team=home_team,
            away_team=away_team,
            predicted_result=predicted_result,
            actual_result=actual_result,
            predicted_probabilities=predicted_probabilities,
            confidence_score=confidence_score,
            key_factors_used=key_factors,
            match_characteristics=match_characteristics,
            was_correct=(predicted_result == actual_result),
            error_magnitude=error_magnitude,
            home_metrics=asdict(home_metrics) if home_metrics else None,
            away_metrics=asdict(away_metrics) if away_metrics else None,
            core_analysis=asdict(core_analysis) if core_analysis else None
        )
        
        self.learning_engine.record_prediction(outcome)
        
        # Update league-specific metrics
        if league not in self.learning_engine.learning_metrics.league_metrics:
            self.learning_engine.learning_metrics.league_metrics[league] = LeagueSpecificMetrics(league_name=league)
        
        league_metrics = self.learning_engine.learning_metrics.league_metrics[league]
        league_metrics.sample_size += 1
        
        # Track league-specific patterns
        if predicted_result == actual_result:
            # Correct prediction - no adjustment needed
            pass
        else:
            # Wrong prediction - adjust league factors
            if league == "Premier League":
                # Premier League specific learning adjustments
                if actual_result == 'D' and predicted_result != 'D':
                    league_metrics.draw_tendency = min(0.35, league_metrics.draw_tendency + 0.01)
                elif actual_result == 'H' and predicted_result == 'A':
                    league_metrics.home_advantage_factor = min(1.3, league_metrics.home_advantage_factor + 0.02)
        
        # Update factor weights periodically
        if self.learning_engine.learning_metrics.total_predictions % 30 == 0:
            self.learning_engine.adjust_factor_weights(MatchFactors())

class EnhancedForebetAnalyzer:
    """Main analyzer class with manual learning capabilities"""
    
    def __init__(self):
        self.factors = MatchFactors()
        self.decision_engine = DecisionEngine()
        self.league_standings = {}
        self.team_cache = {}
        self.adaptive_factors = AdaptiveFactors()
        # Removed auto-verification system - now using manual learning
    
    def fetch_with_playwright_lightweight(self, url: str):
        """Fetch page using Playwright - Optimized version"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-extensions',
                        '--disable-gpu'
                    ]
                )
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )
                
                context.set_default_navigation_timeout(25000)
                context.set_default_timeout(15000)
                
                page = context.new_page()
                
                # Block unnecessary resources
                page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,eot,css}", 
                          lambda route: route.abort())
                
                html = None
                strategies = [
                    {'wait_until': 'domcontentloaded', 'timeout': 10000},
                    {'wait_until': 'load', 'timeout': 15000},
                ]
                
                for strategy in strategies:
                    try:
                        response = page.goto(
                            url,
                            timeout=strategy['timeout'],
                            wait_until=strategy['wait_until']
                        )
                        
                        if response and response.ok:
                            page.wait_for_timeout(1000)
                            html = page.content()
                            break
                    except:
                        continue
                
                browser.close()
                return html
                    
        except Exception as e:
            print(f"   ❌ Playwright error: {str(e)[:50]}")
            return None
    
    def fetch_page_soup(self, url: str):
        """Fetch page with fallback"""
        print("   Fetching with Playwright...")
        html = self.fetch_with_playwright_lightweight(url)
        
        if not html:
            print("   Trying requests fallback...")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                }
                response = requests.get(url, headers=headers, timeout=15)
                html = response.text
            except:
                return None
        
        if not html:
            return None
        
        try:
            return BeautifulSoup(html, "html.parser")
        except:
            return None
    
    def clean_team_name(self, name: str):
        """Clean team name for display"""
        if not name or not isinstance(name, str):
            return ""
        
        original = name
        
        name = re.sub(r'onmouseover\s*=\s*["\']showhint\([^)]*\)["\']', '', name, flags=re.IGNORECASE)
        name = re.sub(r'onclick\s*=\s*["\'][^"\']*["\']', '', name, flags=re.IGNORECASE)
        name = re.sub(r'onmouseout\s*=\s*["\'][^"\']*["\']', '', name, flags=re.IGNORECASE)
        name = re.sub(r'javascript:[^"\'\s]*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'<[^>]+>', '', name)
        name = re.sub(r'showhint\([^)]*\)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*(?:Prediction|Preview|Stats|H2H|Odds|Forebet|Tips|Match).*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'^[\s\'\"\-]+|[\s\'\"\-]+$', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        suffixes = [r'U\d+', 'Reserves', 'B', 'II', 'U23', 'U19', 'U21', 'FC', 'CF', 'SC', 'AFC', 'CFC', 
                   'AC', 'AS', 'SS', 'SV', 'SK', 'FK', 'IF', 'FF', 'CF', 'SD', 'UD', 'CD', 'AD', 'CA', 'EC']
        suffix_pattern = r'\s+(' + '|'.join(suffixes) + r')$'
        name = re.sub(suffix_pattern, '', name, flags=re.IGNORECASE)
        
        name = ' '.join(word for word in name.split() if word)
        
        if not name or len(name) < 2:
            name = re.sub(r'[^A-Za-z0-9\s\-]', '', original)
            name = re.sub(r'\s+', ' ', name).strip()
            if len(name) < 2:
                return original.strip()[:30]
        
        return name.strip()
    
    def clean_team_name_display(self, name: str):
        """Clean team name for display - AGGRESSIVE CLEANING"""
        if not name or not isinstance(name, str):
            return ""
        
        name = self.clean_team_name(name)
        name = re.sub(r'&[a-z]+;', '', name, flags=re.IGNORECASE)
        name = re.sub(r'&lt;.*?&gt;', '', name, flags=re.IGNORECASE)
        name = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', name, flags=re.IGNORECASE)
        name = re.sub(r'javascript:[^"\'\s]*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'[^\w\s\-]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        if len(name) > 20:
            name = name[:17] + '...'
        
        return name
    
    def normalize_name(self, s: str):
        """Normalize team name for comparison"""
        if not s or not isinstance(s, str):
            return ""
        
        s = s.lower()
        s = re.sub(r'\b(the|fc|cf|sc|afc|united|city|town|rovers|athletic|club|dept|ac|as)\b', '', s)
        s = re.sub(r'\b(jong|u23|u21|u19|ii|2|reserves|youth|academy|prediction|preview|stats|h2h)\b', '', s)
        s = re.sub(r'[^\w\s\-]', '', s)
        s = re.sub(r'\d+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        common_words = {'team', 'football', 'soccer', 'sport', 'sports', 'association'}
        words = [word for word in s.split() if word not in common_words and len(word) > 1]
        
        return ' '.join(words)
    
    def extract_team_names_and_positions(self, soup: BeautifulSoup):
        """Extract team names and their league positions from Forebet page"""
        if not soup:
            return None, None, None, None
        
        home_team, away_team = None, None
        home_position, away_position = None, None
        
        header_selectors = [
            'div.rcnt > strong',
            'h1.team-name',
            'div.team_info strong',
            'div.mn_team',
            'span.mn_team',
            'div.headline > strong',
            'div.headline > span'
        ]
        
        for selector in header_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if ' vs ' in text.lower() or ' - ' in text:
                    parts = re.split(r'\s+vs\s+|\s+-\s+', text, flags=re.IGNORECASE)
                    if len(parts) >= 2:
                        home_team = self.clean_team_name(parts[0])
                        away_team = self.clean_team_name(parts[1])
                        break
            if home_team and away_team:
                break
        
        if not home_team or not away_team:
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                patterns = [
                    r'([^\-]+)\s+vs\s+([^\-]+)\s+Prediction',
                    r'([^\-]+)\s+-\s+([^\-]+)\s+Predictions',
                    r'([^\-]+)\s+VS\s+([^\-]+)',
                    r'([^\-]+)\s+vs\s+([^\-]+)\s+-\s+\d'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, title_text, re.IGNORECASE)
                    if match:
                        home_team = self.clean_team_name(match.group(1))
                        away_team = self.clean_team_name(match.group(2))
                        break
        
        if not home_team or not away_team:
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                content = tag.get('content', '')
                if 'vs' in content.lower() and 'prediction' in content.lower():
                    match = re.search(r'([^\-]+)\s+vs\s+([^\-]+)', content, re.IGNORECASE)
                    if match:
                        home_team = self.clean_team_name(match.group(1))
                        away_team = self.clean_team_name(match.group(2))
                        break
        
        standings = self.extract_standings_table(soup)
        
        if home_team:
            home_position = self.get_position_from_standings(standings, home_team)
        if away_team:
            away_position = self.get_position_from_standings(standings, away_team)
        
        for team, pos in standings.items():
            self.league_standings[team] = pos
        
        if home_team and home_team not in self.league_standings and home_position:
            self.league_standings[home_team] = home_position
        if away_team and away_team not in self.league_standings and away_position:
            self.league_standings[away_team] = away_position
        
        return home_team, away_team, home_position, away_position
    
    def extract_standings_table(self, soup: BeautifulSoup):
        """Extract the full standings table from Forebet page"""
        standings = {}
        
        standings_selectors = [
            'table.stage-standing',
            'table.standing_tab',
            'table#standing',
            'table.stats',
            'table.standings',
            'div.stage-standing table',
            'div.standing_tab table'
        ]
        
        for selector in standings_selectors:
            try:
                tables = soup.select(selector)
                for table in tables:
                    text = table.get_text()
                    if any(keyword in text.lower() for keyword in ['pts', 'p', 'w', 'd', 'l', 'gf', 'ga', 'gd', '+/-', 'position', 'rank']):
                        rows = table.find_all("tr")
                        for row in rows[1:]:
                            cols = row.find_all("td")
                            if len(cols) >= 3:
                                pos_text = cols[0].get_text(strip=True)
                                if pos_text.isdigit():
                                    position = int(pos_text)
                                    team_name = None
                                    for col in cols[1:]:
                                        col_text = col.get_text(strip=True)
                                        if col_text and not col_text.isdigit() and len(col_text) > 2:
                                            if len(col_text.split()) > 1 or len(col_text) > 4:
                                                team_name = self.clean_team_name(col_text)
                                                if team_name:
                                                    break
                                    
                                    if team_name:
                                        standings[team_name] = position
                        
                        if standings:
                            return standings
            except Exception as e:
                continue
        
        html_text = str(soup)
        patterns = [
            r'(\d+)\.\s+([A-Za-z\s\-\.]+?)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+[\+\-]?\d+',
            r'(\d+)\s+([A-Za-z\s\-\.]+?)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+[\+\-]?\d+',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match) >= 2:
                    try:
                        position = int(match[0])
                        team_name = self.clean_team_name(match[1])
                        if team_name and len(team_name) > 2:
                            standings[team_name] = position
                    except:
                        continue
        
        return standings
    
    def get_position_from_standings(self, standings: dict, team_name: str):
        """Get position for a specific team from standings"""
        if not standings or not team_name:
            return None
        
        team_norm = self.normalize_name(team_name)
        
        for team, pos in standings.items():
            if self.normalize_name(team) == team_norm:
                return pos
        
        for team, pos in standings.items():
            team_norm_db = self.normalize_name(team)
            
            if team_norm in team_norm_db or team_norm_db in team_norm:
                return pos
            
            team_words = set(team_norm.split())
            db_words = set(team_norm_db.split())
            common_words = team_words.intersection(db_words)
            
            if len(common_words) >= 2 or (len(common_words) == 1 and len(list(common_words)[0]) > 4):
                return pos
        
        for team, pos in standings.items():
            team_norm_db = self.normalize_name(team)
            if team_norm[:6] == team_norm_db[:6] or team_norm[-6:] == team_norm_db[-6:]:
                return pos
        
        return None
    
    def extract_head_to_head_matches(self, soup: BeautifulSoup, home_team: str, away_team: str) -> List[H2HMatch]:
        """Extract head-to-head matches from Forebet page"""
        h2h_matches = []
        
        home_norm = self.normalize_name(home_team)
        away_norm = self.normalize_name(away_team)
        
        h2h_sections = []
        
        for tag in soup.find_all(['h2', 'h3', 'h4', 'div', 'section']):
            text = (tag.get_text() or "").lower()
            if 'head to head' in text or 'h2h' in text or 'previous meetings' in text:
                h2h_sections.append(tag)
        
        if not h2h_sections:
            h2h_sections = [soup]
        
        patterns = [
            r'(\d{2}/\d{2}/\d{4})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+?)(?:\s+([A-Za-z0-9]{2,6}))?(?=\s|$|<)',
            r'(\d{2}/\d{2}/\d{2})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            r'(\d{2}/\d{2})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            r'([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
        ]
        
        for container in h2h_sections:
            text = container.get_text(separator='  ', strip=True)
            
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        groups = m.groups()
                        
                        if len(groups) >= 4:
                            if re.match(r'\d{2}/\d{2}', groups[0]):
                                if '/' in groups[0] and len(groups[0].split('/')) == 3:
                                    date_str = groups[0]
                                elif '/' in groups[0] and len(groups[0].split('/')) == 2:
                                    date_str = f"{groups[0]}/{datetime.now().year}"
                                else:
                                    continue
                                
                                team1 = self.clean_team_name(groups[1])
                                score1 = int(groups[2])
                                score2 = int(groups[3])
                                team2 = self.clean_team_name(groups[4])
                                comp = groups[5] if len(groups) > 5 and groups[5] else "Unknown"
                            else:
                                date_str = "Recent"
                                team1 = self.clean_team_name(groups[0])
                                score1 = int(groups[1])
                                score2 = int(groups[2])
                                team2 = self.clean_team_name(groups[3])
                                comp = "Unknown"
                            
                            t1n = self.normalize_name(team1)
                            t2n = self.normalize_name(team2)
                            
                            match_found = False
                            
                            if (home_norm in t1n or t1n in home_norm) and (away_norm in t2n or t2n in away_norm):
                                match_found = True
                            elif (home_norm in t2n or t2n in home_norm) and (away_norm in t1n or t1n in away_norm):
                                match_found = True
                            else:
                                home_words = set(home_norm.split())
                                away_words = set(away_norm.split())
                                t1_words = set(t1n.split())
                                t2_words = set(t2n.split())
                                
                                home_t1_overlap = len(home_words.intersection(t1_words))
                                home_t2_overlap = len(home_words.intersection(t2_words))
                                away_t1_overlap = len(away_words.intersection(t1_words))
                                away_t2_overlap = len(away_words.intersection(t2_words))
                                
                                if (home_t1_overlap >= 2 and away_t2_overlap >= 2) or \
                                   (home_t2_overlap >= 2 and away_t1_overlap >= 2):
                                    match_found = True
                            
                            if match_found:
                                h2h_matches.append(H2HMatch(
                                    date=date_str,
                                    home_team=team1,
                                    away_team=team2,
                                    home_goals=score1,
                                    away_goals=score2,
                                    competition=comp
                                ))
                    
                    except (ValueError, IndexError, AttributeError):
                        continue
        
        seen = set()
        unique = []
        for m in h2h_matches:
            key = (m.date, m.home_team, m.away_team, m.home_goals, m.away_goals)
            if key not in seen:
                seen.add(key)
                unique.append(m)
        
        unique.sort(key=lambda x: self.parse_date_to_sortable(x.date), reverse=True)
        
        return unique[:15]
    
    def calculate_h2h_metrics(self, h2h_matches: List[H2HMatch], home_team: str, away_team: str) -> H2HMetrics:
        """Calculate H2H metrics from match data"""
        
        metrics = H2HMetrics()
        
        if not h2h_matches:
            return metrics
        
        home_norm = self.normalize_name(home_team)
        away_norm = self.normalize_name(away_team)
        
        for match in h2h_matches:
            home_norm_match = self.normalize_name(match.home_team)
            away_norm_match = self.normalize_name(match.away_team)
            
            metrics.total_matches += 1
            
            is_home_in_match_home = (home_norm in home_norm_match or 
                                    any(word in home_norm_match for word in home_norm.split() if len(word) > 2))
            
            if match.home_goals > match.away_goals:
                if is_home_in_match_home:
                    metrics.home_wins += 1
                    metrics.recent_results.append('H')
                else:
                    metrics.away_wins += 1
                    metrics.recent_results.append('A')
            elif match.home_goals < match.away_goals:
                if is_home_in_match_home:
                    metrics.away_wins += 1
                    metrics.recent_results.append('A')
                else:
                    metrics.home_wins += 1
                    metrics.recent_results.append('H')
            else:
                metrics.draws += 1
                metrics.recent_results.append('D')
            
            if is_home_in_match_home:
                metrics.home_goals_for += match.home_goals
                metrics.home_goals_against += match.away_goals
                metrics.away_goals_for += match.away_goals
                metrics.away_goals_against += match.home_goals
            else:
                metrics.home_goals_for += match.away_goals
                metrics.home_goals_against += match.home_goals
                metrics.away_goals_for += match.home_goals
                metrics.away_goals_against += match.away_goals
            
            metrics.avg_total_goals += (match.home_goals + match.away_goals)
        
        if metrics.total_matches > 0:
            metrics.draw_rate = metrics.draws / metrics.total_matches
            metrics.home_win_rate = metrics.home_wins / metrics.total_matches
            metrics.away_win_rate = metrics.away_wins / metrics.total_matches
            metrics.avg_total_goals = metrics.avg_total_goals / metrics.total_matches
            
            metrics.home_goals_for = metrics.home_goals_for / metrics.total_matches
            metrics.home_goals_against = metrics.home_goals_against / metrics.total_matches
            metrics.away_goals_for = metrics.away_goals_for / metrics.total_matches
            metrics.away_goals_against = metrics.away_goals_against / metrics.total_matches
            
            if len(metrics.recent_results) >= 3:
                recent = metrics.recent_results[:3]
                recent_home_wins = sum(1 for r in recent if r == 'H')
                recent_away_wins = sum(1 for r in recent if r == 'A')
                metrics.recent_trend = (recent_home_wins - recent_away_wins) / 3.0
        
        return metrics
    
# Removed extract_home_away_from_symbols - was extracting incorrect data
    def extract_matches_from_section(self, html: str, team_name: str, venue: str) -> List[Match]:
        """Extract matches from specific HTML section (home or away)"""
        matches = []
        clean_team = self.clean_team_name(team_name)
        team_normalized = self.normalize_name(clean_team)
        
        # Pattern for match rows like in the image
        pattern = r'(\d{2}/\d{2}\s+\d{4})\s+([^\(]+?)\s+\((\d+)\s*-\s*(\d+)\)\s+(\d+)\s*-\s*(\d+)\s+([^\s]+)'
        
        for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
            try:
                date, teams_str, ht_half1, ht_half2, ft_half1, ft_half2, competition = match.groups()
                
                # Clean up team names
                teams = teams_str.strip()
                
                # Parse the teams - format is usually "Team1 - Team2" or "Team1 Team2"
                if ' - ' in teams:
                    team1, team2 = teams.split(' - ', 1)
                else:
                    # Try to split by common separators
                    team1 = teams.split()[0] if teams.split() else ""
                    team2 = ' '.join(teams.split()[1:]) if len(teams.split()) > 1 else ""
                
                team1_clean = self.clean_team_name(team1.strip())
                team2_clean = self.clean_team_name(team2.strip())
                
                # Determine if our team is home or away in this match
                is_home_in_match = self.normalize_name(team1_clean) == team_normalized
                
                # Determine result based on venue and score
                if is_home_in_match:
                    home_goals = int(ft_half1)
                    away_goals = int(ft_half2)
                    opponent = team2_clean
                else:
                    home_goals = int(ft_half2)
                    away_goals = int(ft_half1)
                    opponent = team1_clean
                
                # Determine result for our team
                if (is_home_in_match and home_goals > away_goals) or (not is_home_in_match and away_goals > home_goals):
                    result = 'W'
                elif home_goals == away_goals:
                    result = 'D'
                else:
                    result = 'L'
                
                # Determine if this match is actually home or away for our team
                actual_venue = 'H' if (venue == 'H' and is_home_in_match) or (venue == 'A' and not is_home_in_match) else 'A'
                
                # Get opponent position
                opponent_position = self.get_position_from_standings(self.league_standings, opponent)
                
                match_obj = Match(
                    date=date.strip(),
                    home_team=team1_clean,
                    away_team=team2_clean,
                    score=f"{ft_half1}-{ft_half2}",
                    result=result,
                    goals_scored=home_goals if is_home_in_match else away_goals,
                    goals_conceded=away_goals if is_home_in_match else home_goals,
                    venue=actual_venue,
                    competition=competition.strip(),
                    opponent_position=opponent_position
                )
                
                matches.append(match_obj)
                
            except (ValueError, IndexError, AttributeError) as e:
                continue
        
        return matches
    
    def get_last_matches(self, html: str, team_name: str, competition_name: str = None, count: int = 8):
        """Extract recent matches using improved regex patterns"""
        if not html or not team_name:
            return []
        
        matches = []
        clean_team = self.clean_team_name(team_name)
        team_normalized = self.normalize_name(clean_team)
        
        if not team_normalized:
            return []
        
        cache_key = f"{team_normalized}_{hash(html[:1000])}"
        if cache_key in self.team_cache:
            return self.team_cache[cache_key][:count]
        
        patterns = [
            r'(\d{2}/\d{2}/\d{4})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            r'(\d{2}/\d{2}/\d{2})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            r'([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            r'([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+?)\s*\(([^)]+)\)',
        ]
        
        all_found_matches = []
        
        for pattern in patterns:
            found_matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
            all_found_matches.extend(found_matches)
        
        for match_tuple in all_found_matches:
            try:
                if len(match_tuple) == 5:
                    if re.match(r'\d{2}/\d{2}', match_tuple[0]):
                        date, team1, g1, g2, team2 = match_tuple
                        competition = competition_name or "Unknown"
                    else:
                        team1, g1, g2, team2, competition = match_tuple
                        date = ""
                elif len(match_tuple) == 4:
                    team1, g1, g2, team2 = match_tuple
                    date = ""
                    competition = competition_name or "Unknown"
                else:
                    continue
                
                team1_clean = self.clean_team_name(team1.strip())
                team2_clean = self.clean_team_name(team2.strip())
                
                if not team1_clean or not team2_clean:
                    continue
                
                try:
                    g1_int = int(g1.strip())
                    g2_int = int(g2.strip())
                except ValueError:
                    continue
                
                team1_norm = self.normalize_name(team1_clean)
                team2_norm = self.normalize_name(team2_clean)
                
                opponent_position = None
                
                if team_normalized in team1_norm or team1_norm in team_normalized:
                    result = "W" if g1_int > g2_int else "D" if g1_int == g2_int else "L"
                    venue = 'H'
                    goals_scored = g1_int
                    goals_conceded = g2_int
                    opponent = team2_clean
                    opponent_position = self.get_position_from_standings(self.league_standings, team2_clean)
                    
                    match_obj = Match(
                        date=date.strip() if date else "",
                        home_team=team1_clean,
                        away_team=team2_clean,
                        score=f"{g1_int}-{g2_int}",
                        result=result,
                        goals_scored=goals_scored,
                        goals_conceded=goals_conceded,
                        venue=venue,
                        competition=competition.strip(),
                        opponent_position=opponent_position
                    )
                    matches.append(match_obj)
                    
                elif team_normalized in team2_norm or team2_norm in team_normalized:
                    result = "W" if g2_int > g1_int else "D" if g2_int == g1_int else "L"
                    venue = 'A'
                    goals_scored = g2_int
                    goals_conceded = g1_int
                    opponent = team1_clean
                    opponent_position = self.get_position_from_standings(self.league_standings, team1_clean)
                    
                    match_obj = Match(
                        date=date.strip() if date else "",
                        home_team=team1_clean,
                        away_team=team2_clean,
                        score=f"{g1_int}-{g2_int}",
                        result=result,
                        goals_scored=goals_scored,
                        goals_conceded=goals_conceded,
                        venue=venue,
                        competition=competition.strip(),
                        opponent_position=opponent_position
                    )
                    matches.append(match_obj)
                    
            except Exception as e:
                continue
        
        matches.sort(key=lambda x: self.parse_date_to_sortable(x.date), reverse=True)
        
        unique_matches = []
        seen = set()
        for match in matches:
            key = (match.date, match.home_team, match.away_team, match.score)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        self.team_cache[cache_key] = unique_matches
        
        return unique_matches[:count]
    
    def parse_date_to_sortable(self, date_str: str):
        """Parse date to sortable format"""
        if not date_str:
            return '00000000'
        
        date_patterns = [
            r'(\d{2})/(\d{2})/(\d{4})',
            r'(\d{2})/(\d{2})/(\d{2})',
            r'(\d{2})/(\d{2})',
        ]
        
        for pattern in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    day, month, year = groups
                    if len(year) == 2:
                        year = '20' + year
                    return f"{year}{month.zfill(2)}{day.zfill(2)}"
                elif len(groups) == 2:
                    day, month = groups
                    current_year = datetime.now().year
                    return f"{current_year}{month.zfill(2)}{day.zfill(2)}"
        
        return '00000000'
    
    def extract_notable_results(self, matches: List[Match], team_name: str) -> List[str]:
        """Extract notable wins/losses/draws from recent matches - IMPROVED"""
        notable = []
        
        if not matches:
            return ["No recent matches"]
        
        for match in matches[:6]:  # Last 6 matches max
            # Determine opponent team
            if match.venue == 'H':
                opponent_team = match.away_team
                opponent_clean = self.clean_team_name_display(opponent_team)
            else:
                opponent_team = match.home_team
                opponent_clean = self.clean_team_name_display(opponent_team)
            
            # Clean opponent name for display (limit length)
            opponent_display = opponent_clean[:15] if len(opponent_clean) > 15 else opponent_clean
            
            # Check opponent position
            is_top6_opponent = match.opponent_position and match.opponent_position <= 6
            is_bottom6_opponent = False
            if match.opponent_position:
                total_teams = 20  # Assuming 20-team league
                is_bottom6_opponent = match.opponent_position >= (total_teams - 5)
            
            # Determine result
            if match.result == 'W':
                if is_top6_opponent:
                    notable.append(f"✅ Beat #{match.opponent_position} {opponent_display} {match.score}")
                elif match.goals_scored >= 4:
                    notable.append(f"🔥 Big win: {match.score} vs {opponent_display}")
                elif match.goals_scored - match.goals_conceded >= 3:
                    notable.append(f"✅ Comfortable win: {match.score} vs {opponent_display}")
                elif is_bottom6_opponent:
                    notable.append(f"✅ Win vs #{match.opponent_position} {opponent_display} {match.score}")
                else:
                    notable.append(f"✅ Win: {match.score} vs {opponent_display}")
                    
            elif match.result == 'D':
                if is_top6_opponent:
                    notable.append(f"⚪ Good draw: {match.score} vs #{match.opponent_position} {opponent_display}")
                elif is_bottom6_opponent:
                    notable.append(f"⚠️ Poor draw: {match.score} vs #{match.opponent_position} {opponent_display}")
                else:
                    notable.append(f"⚪ Draw: {match.score} vs {opponent_display}")
                    
            elif match.result == 'L':
                if is_top6_opponent:
                    notable.append(f"❌ Lost to #{match.opponent_position} {opponent_display} {match.score}")
                elif match.goals_conceded >= 4:
                    notable.append(f"💥 Heavy loss: {match.score} vs {opponent_display}")
                elif is_bottom6_opponent:
                    notable.append(f"❌ Bad loss to #{match.opponent_position} {opponent_display} {match.score}")
                else:
                    notable.append(f"❌ Loss: {match.score} vs {opponent_display}")
        
        # If we don't have notable results, create a summary
        if not notable:
            wins = sum(1 for m in matches if m.result == 'W')
            draws = sum(1 for m in matches if m.result == 'D')
            losses = sum(1 for m in matches if m.result == 'L')
            
            if wins >= 4:
                notable.append(f"🔥 Excellent: {wins} wins in last {len(matches)}")
            elif wins >= 2 and losses <= 1:
                notable.append(f"✅ Good form: {wins}W-{draws}D-{losses}L")
            elif losses >= 4:
                notable.append(f"💥 Poor form: {losses} losses in last {len(matches)}")
            elif draws >= 4:
                notable.append(f"⚪ Draw specialist: {draws} draws")
            else:
                notable.append(f"Form: {wins}W-{draws}D-{losses}L")
        
        return notable[:3]  # Return max 3 notable results
    
    def extract_competition_name(self, soup: BeautifulSoup):
        """Extract competition name from Forebet page"""
        if not soup:
            return "Unknown Competition"
        
        breadcrumb_selectors = [
            'nav.breadcrumb a',
            'div.breadcrumb a',
            '.breadcrumbs a',
            'ul.breadcrumb li a',
            'div.breadcrumbs a'
        ]
        
        for selector in breadcrumb_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and len(text) > 3:
                    text_lower = text.lower()
                    if text_lower not in ['home', 'football', 'predictions', 'matches', 'forebet', 'tips', 'stats']:
                        if any(keyword in text_lower for keyword in ['league', 'cup', 'championship', 'division', 
                                                                    'liga', 'premier', 'serie', 'bundes', 'lique', 
                                                                    'eredivisie', 'primeira', 'super lig']):
                            return text
        
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text()
            patterns = [
                r'Prediction\s*[-\–]\s*([^\-]+)',
                r'-\s*([^-]+)\s+Prediction',
                r'([A-Za-z\s]+)\s+-\s+\d',
                r'([^\-]+)\s+vs\s+[^\-]+\s+-\s+([^\-]+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, title_text, re.IGNORECASE)
                if match:
                    comp = match.group(1).strip()
                    if comp and len(comp) > 3:
                        return comp
        
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for heading in headings:
            text = heading.get_text(strip=True)
            if text and any(keyword in text.lower() for keyword in ['league', 'cup', 'championship', 'division']):
                return text
        
        page_text = soup.get_text()
        competition_patterns = [
            r'Competition:\s*([^\n]+)',
            r'League:\s*([^\n]+)',
            r'Tournament:\s*([^\n]+)'
        ]
        
        for pattern in competition_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                comp = match.group(1).strip()
                if comp and len(comp) > 3:
                    return comp
        
        return "Unknown Competition"
    
    def determine_match_type(self, soup: BeautifulSoup, competition: str) -> MatchType:
        """Determine the type of match (league, cup, etc.)"""
        comp_lower = competition.lower()
        page_text = soup.get_text().lower()
        
        if 'cup' in comp_lower or 'trophy' in comp_lower or 'champions league' in page_text or 'europa' in page_text:
            return MatchType.CUP
        elif 'friendly' in comp_lower or 'friendly' in page_text:
            return MatchType.FRIENDLY
        elif 'playoff' in comp_lower or 'play-off' in page_text:
            return MatchType.PLAYOFF
        elif 'league' in comp_lower or 'division' in comp_lower or 'championship' in comp_lower:
            return MatchType.LEAGUE
        else:
            return MatchType.UNKNOWN
    
    def calculate_goal_probabilities(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics):
        """Calculate goal market probabilities with improved logic"""
        
        home_expected = (home_metrics.home_scoring_rate * 0.7 + 
                        (2 - away_metrics.away_conceding_rate) * 0.3)
        
        away_expected = (away_metrics.away_scoring_rate * 0.7 + 
                        (2 - home_metrics.home_conceding_rate) * 0.3)
        
        home_expected *= (0.8 + home_metrics.recent_form_rating * 0.4)
        away_expected *= (0.8 + away_metrics.recent_form_rating * 0.4)
        
        home_expected *= (1 + home_metrics.goal_trend * 0.3)
        away_expected *= (1 + away_metrics.goal_trend * 0.3)
        
        home_expected *= (1 - home_metrics.defense_trend * 0.2)
        away_expected *= (1 - away_metrics.defense_trend * 0.2)
        
        if home_metrics.h2h_metrics.total_matches >= 3:
            h2h_weight = 0.15
            h2h_total_goals = home_metrics.h2h_metrics.avg_total_goals
            
            if h2h_total_goals > 2.8:
                home_expected = home_expected * (1 - h2h_weight) + (h2h_total_goals / 2) * h2h_weight
                away_expected = away_expected * (1 - h2h_weight) + (h2h_total_goals / 2) * h2h_weight
            elif h2h_total_goals < 1.2:
                home_expected = home_expected * (1 - h2h_weight) + (h2h_total_goals / 2) * h2h_weight
                away_expected = away_expected * (1 - h2h_weight) + (h2h_total_goals / 2) * h2h_weight
        
        home_expected = max(0.1, min(4.0, home_expected))
        away_expected = max(0.1, min(3.5, away_expected))
        
        def poisson_prob(lambda_, k):
            if k > 8:
                return 0
            return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)
        
        over_25_prob = 0
        bts_prob = 0
        
        max_goals = 6
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson_prob(home_expected, i) * poisson_prob(away_expected, j)
                if i + j > 2.5:
                    over_25_prob += prob
                if i > 0 and j > 0:
                    bts_prob += prob
        
        home_clean_sheet = math.exp(-away_expected)
        away_clean_sheet = math.exp(-home_expected)
        
        over_15_prob = 0
        over_35_prob = 0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson_prob(home_expected, i) * poisson_prob(away_expected, j)
                if i + j > 1.5:
                    over_15_prob += prob
                if i + j > 3.5:
                    over_35_prob += prob
        
        return {
            'over_15': over_15_prob * 100,
            'over_25': over_25_prob * 100,
            'over_35': over_35_prob * 100,
            'under_25': (1 - over_25_prob) * 100,
            'bts': bts_prob * 100,
            'expected_home_goals': home_expected,
            'expected_away_goals': away_expected,
            'total_expected_goals': home_expected + away_expected,
            'home_clean_sheet': home_clean_sheet * 100,
            'away_clean_sheet': away_clean_sheet * 100
        }
    
    def display_analysis(self, home_team: str, away_team: str,
                        home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                        probabilities: Dict, goal_markets: Dict,
                        competition: str, home_matches: List[Match], away_matches: List[Match],
                        decisions: Dict, core_analysis: CoreAnalysis, match_summary: MatchSummary,
                        h2h_matches: List[H2HMatch], factors: MatchFactors, html: str = ""):
        """Display comprehensive analysis with improved formatting"""
        
        print(f"\n{'⚽ FOOTBALL MATCH ANALYZER ⚽'.center(70)}")
        print("─" * 70)
        
        home_display = self.clean_team_name_display(home_team)
        away_display = self.clean_team_name_display(away_team)
        
        print(f"\n🏆 {competition}")
        print(f"🏠 {home_display} (#{home_metrics.league_position if home_metrics.league_position else 'N/A'})")
        print(f"🆚 {away_display} (#{away_metrics.league_position if away_metrics.league_position else 'N/A'})")
        
        # MAIN RECOMMENDATION - MOVED TO TOP
        print(f"\n{'🎯 RECOMMENDATION'.center(70, '═')}")
        
        if decisions.get('primary_bet'):
            primary = decisions['primary_bet']
            selection_map = {
                'HOME_WIN': f"🏠 {home_display[:15]} WIN",
                'AWAY_WIN': f"🛣️ {away_display[:15]} WIN",
                'DRAW': '🤝 DRAW'
            }
            
            selection = selection_map.get(primary['selection'], primary['selection'])
            
            conf_icons = {
                'VERY HIGH': '🟢🟢🟢',
                'HIGH': '🟢🟢⚪',
                'MEDIUM': '🟡🟡⚪',
                'LOW': '🔴⚪⚪'
            }
            conf_icon = conf_icons.get(decisions['confidence'], '⚪⚪⚪')
            
            print(f"\n✅ BET: {selection}")
            print(f"📊 Confidence: {conf_icon} {decisions['confidence']} ({decisions['confidence_score']}/15)")
            print(f"💰 Stake: {primary['stake_suggestion']}")
            print(f"🎲 Odds Range: {primary['odds_range']}")
            print(f"📈 Probability: {primary.get('probability', 0):.1f}%")
            
            # Add narrative explanation
            if primary['selection'] == 'HOME_WIN':
                print(f"📝 Why Home Win: {home_display[:12]} has clear advantages at home")
            elif primary['selection'] == 'AWAY_WIN':
                print(f"📝 Why Away Win: {away_display[:12]} shows superior form and quality")
            elif primary['selection'] == 'DRAW':
                print(f"📝 Why Draw: Teams are evenly matched with strong draw tendencies")
            
            if primary.get('detailed_reasoning'):
                reasons = primary['detailed_reasoning'].split(' | ')[:3]
                print(f"💡 Key Factors: {' • '.join(reasons)}")
        else:
            print(f"\n❌ NO BET RECOMMENDED")
            print(f"📊 Confidence: {decisions.get('confidence', 'LOW')} ({decisions.get('confidence_score', 0)}/15)")
            print(f"⚠️ Risk: {decisions.get('risk_level', 'HIGH')}")
            
            # Add narrative for no bet
            if decisions.get('confidence_score', 0) < 6:
                print(f"📝 Why No Bet: Too many conflicting factors and low confidence")
            elif decisions.get('risk_level') == 'VERY HIGH':
                print(f"📝 Why No Bet: Match too unpredictable, high risk of loss")
            else:
                print(f"📝 Why No Bet: No clear value or advantage identified")
        
        print("═" * 70)
        
        # H2H Section
        if h2h_matches:
            print(f"🤝 HEAD-TO-HEAD HISTORY")
            print(f"Total Matches: {home_metrics.h2h_metrics.total_matches}")
            print(f"Record: {home_display[:15]} {home_metrics.h2h_metrics.home_wins}W – {home_metrics.h2h_metrics.draws}D – {home_metrics.h2h_metrics.away_wins}L")
            print(f"Win Rates: Home {home_metrics.h2h_metrics.home_win_rate:.0%} | Away {home_metrics.h2h_metrics.away_win_rate:.0%} | Draw {home_metrics.h2h_metrics.draw_rate:.0%}")
            
            if home_metrics.h2h_metrics.recent_trend != 0:
                trend = "favors HOME" if home_metrics.h2h_metrics.recent_trend > 0 else "favors AWAY"
                print(f"Recent Trend: {trend}")
            
            if h2h_matches:
                print(f"Recent H2H Results (last 3):")
                for i, match in enumerate(h2h_matches[:3], 1):
                    home_clean = self.clean_team_name_display(match.home_team)
                    away_clean = self.clean_team_name_display(match.away_team)
                    
                    result_icon = '✅' if (match.home_goals > match.away_goals and home_team[:10] in match.home_team) or \
                                         (match.away_goals > match.home_goals and home_team[:10] in match.away_team) \
                                         else '❌' if (match.home_goals > match.away_goals and away_team[:10] in match.home_team) or \
                                         (match.away_goals > match.home_goals and away_team[:10] in match.away_team) \
                                         else '⚪'
                    print(f"  {result_icon} {match.date} – {home_clean[:12]} {match.home_goals}-{match.away_goals} {away_clean[:12]}")
            print()
        
        # Recent Form Analysis - IMPROVED FORMATTING
        print(f"📊 RECENT FORM ANALYSIS")
        print(f"{'Team':<18} {'Last 6':<8} {'Key Performances'}")
        print("-" * 80)
        
        # Home team results
        home_results = []
        for i, match in enumerate(home_matches[:6]):
            if i < len(home_matches):
                if home_matches[i].result == 'W':
                    home_results.append('✅')
                elif home_matches[i].result == 'D':
                    home_results.append('⚪')
                else:
                    home_results.append('❌')
            else:
                home_results.append('·')
        
        home_notable = self.extract_notable_results(home_matches, home_team)
        home_notable_str = ' | '.join(home_notable[:2]) if home_notable else "No notable results"
        
        # Ensure full display without truncation
        if len(home_notable_str) > 55:
            home_notable_str = home_notable[0] if home_notable else "Recent matches"
        
        print(f"{home_display[:16]:<18} {''.join(home_results):<8} {home_notable_str}")
        
        # Away team results  
        away_results = []
        for i, match in enumerate(away_matches[:6]):
            if i < len(away_matches):
                if away_matches[i].result == 'W':
                    away_results.append('✅')
                elif away_matches[i].result == 'D':
                    away_results.append('⚪')
                else:
                    away_results.append('❌')
            else:
                away_results.append('·')
        
        away_notable = self.extract_notable_results(away_matches, away_team)
        away_notable_str = ' | '.join(away_notable[:2]) if away_notable else "No notable results"
        
        # Ensure full display without truncation
        if len(away_notable_str) > 55:
            away_notable_str = away_notable[0] if away_notable else "Recent matches"
        
        print(f"{away_display[:16]:<18} {''.join(away_results):<8} {away_notable_str}")
        print()
        
        # Key Factors
        print(f"🎯 KEY MATCH FACTORS")
        factors_list = match_summary.characteristics[:6]
        if core_analysis.detected_patterns:
            for i, pattern in enumerate(core_analysis.detected_patterns[:2]):
                factors_list.insert(0, f"PATTERN: {pattern}")
        
        for i, factor in enumerate(factors_list, 1):
            clean_factor = re.sub(r'&[a-z]+;', '', factor)
            clean_factor = re.sub(r'<[^>]+>', '', clean_factor)
            print(f"{i}. {clean_factor[:50]}{'...' if len(clean_factor) > 50 else ''}")
        print()
        
        # Probability Spread
        print(f"📈 PROBABILITY ANALYSIS")
        home_prob = probabilities['home_win']
        away_prob = probabilities['away_win']
        draw_prob = probabilities['draw']
        
        def create_bar(percentage, width=20):
            filled = int(percentage * width / 100)
            return '█' * filled + '░' * (width - filled)
        
        print(f"{home_display[:12]:<12} {create_bar(home_prob)} {home_prob:5.1f}%")
        print(f"{'Draw':<12} {create_bar(draw_prob)} {draw_prob:5.1f}%")
        print(f"{away_display[:12]:<12} {create_bar(away_prob)} {away_prob:5.1f}%")
        print()
        
        # GOAL MARKETS - SIMPLIFIED
        print(f"⚽ GOAL PREDICTIONS")
        print(f"Expected Goals: {home_display[:10]} {goal_markets['expected_home_goals']:.1f} - {away_display[:10]} {goal_markets['expected_away_goals']:.1f}")
        print(f"Total Expected: {goal_markets['total_expected_goals']:.1f} goals")
        
        over_25_icon = '🔥' if goal_markets['over_25'] > 70 else '❄️' if goal_markets['over_25'] < 30 else '⚖️'
        bts_icon = '✅' if goal_markets['bts'] > 70 else '❌' if goal_markets['bts'] < 30 else '⚖️'
        
        print(f"{over_25_icon} Over 2.5 Goals: {goal_markets['over_25']:.0f}%")
        print(f"{bts_icon} Both Teams Score: {goal_markets['bts']:.0f}%")
        print()
        
        # KEY STATS - SIMPLIFIED
        print(f"\n📊 KEY STATS")
        print(f"{'Metric':<20} {'Home':<12} {'Away':<12}")
        print("-" * 50)
        
        # Calculate home/away form from last 6 matches
        home_home_form = 0.0
        away_away_form = 0.0
        
        # Home team: calculate home form from their home matches in last 6
        home_home_matches = [m for m in home_matches[:6] if m.venue == 'H']
        if home_home_matches:
            home_home_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in home_home_matches)
            home_home_form = home_home_points / (len(home_home_matches) * 3)
        
        # Away team: calculate away form from their away matches in last 6
        away_away_matches = [m for m in away_matches[:6] if m.venue == 'A']
        if away_away_matches:
            away_away_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in away_away_matches)
            away_away_form = away_away_points / (len(away_away_matches) * 3)
        
        key_metrics = [
            ('Recent Form', f"{home_metrics.recent_form_rating:.1%}", f"{away_metrics.recent_form_rating:.1%}"),
            ('Goals/Match', f"{home_metrics.goal_scoring_rate:.1f}", f"{away_metrics.goal_scoring_rate:.1f}"),
            ('Home/Away Form', f"{home_home_form:.0%}", f"{away_away_form:.0%}"),
            ('Clean Sheets', f"{home_metrics.clean_sheet_rate:.0%}", f"{away_metrics.clean_sheet_rate:.0%}"),
        ]
        
        for name, home_val, away_val in key_metrics:
            print(f"{name:<20} {home_val:<12} {away_val:<12}")
        
        print()
        
        # Factor Weights (for transparency)
        print(f"⚖️  MODEL FACTOR WEIGHTS (Learning Adjusted)")
        weight_data = [
            ('Form', factors.form_weight),
            ('Recent Form', factors.recent_form_weight),
            ('Attack vs Defense', factors.attack_vs_defense_weight),
            ('Opponent Quality', factors.opponent_quality_weight),
            ('Home Advantage', factors.home_advantage_weight),
            ('H2H', factors.h2h_weight),
            ('Consistency', factors.consistency_weight),
        ]
        
        for name, weight in weight_data:
            bar = '█' * int(weight * 50) + '░' * (50 - int(weight * 50))
            print(f"{name:<18} {bar[:20]} {weight:.2f}")
        print()
        
        # Remove the old betting recommendation section since it's now at the top
        
        # Secondary Bets - SIMPLIFIED
        if decisions.get('secondary_bets'):
            print(f"🎯 OTHER BETS")
            for bet in decisions['secondary_bets'][:2]:
                conf_icon = '🟢' if bet.get('confidence', 'MEDIUM') in ['HIGH', 'MEDIUM-HIGH'] else '🟡'
                print(f"   {conf_icon} {bet.get('selection', 'Unknown')}: {bet.get('probability', 0):.0f}%")
            print()
        
        # Value Bets - FILTERED: Only show if consistent with primary bet
        consistent_value_bets = []
        if decisions.get('value_bets') and decisions.get('primary_bet'):
            primary_selection = decisions['primary_bet'].get('selection', '')
            
            # Map primary bet selections to value bet outcomes
            selection_to_outcome = {
                'HOME_WIN': 'home_win',
                'AWAY_WIN': 'away_win', 
                'DRAW': 'draw'
            }
            
            primary_outcome = selection_to_outcome.get(primary_selection)
            
            # Only include value bets that match the primary recommendation
            for bet in decisions['value_bets']:
                if bet.get('outcome') == primary_outcome:
                    consistent_value_bets.append(bet)
        
        if consistent_value_bets:
            print(f"💎 VALUE OPPORTUNITIES")
            for bet in consistent_value_bets[:2]:
                outcome_map = {'home_win': f'{home_display[:12]} Win', 
                              'away_win': f'{away_display[:12]} Win', 'draw': 'Draw'}
                outcome_name = outcome_map.get(bet.get('outcome', ''), bet.get('outcome', ''))
                print(f"   💰 {outcome_name}: {bet.get('value_percentage', 0):.1f}% value")
            print()
        
        # WARNINGS - SIMPLIFIED
        if decisions.get('avoid_bets'):
            print(f"⚠️ WARNINGS")
            for warning in decisions['avoid_bets'][:3]:
                clean_warning = re.sub(r'&[a-z]+;|<[^>]+>', '', warning)
                if len(clean_warning) > 60:
                    clean_warning = clean_warning[:57] + "..."
                print(f"   • {clean_warning}")
            print()
        
        print("─" * 70)
        print(" 📊 ANALYSIS COMPLETE - BET RESPONSIBLY 📊 ".center(70))
        print("─" * 70)
        print("📚 MANUAL LEARNING: To improve predictions, run with result:")
        print("   python hw.py <url> \"H 3-2\"  (Home win 3-2)")
        print("   python hw.py <url> \"A 1-0\"  (Away win 1-0)")  
        print("   python hw.py <url> \"D 2-2\"  (Draw 2-2)")
        print("─" * 70)
        
        return True
    
    def generate_match_id(self, url: str) -> str:
        """Generate unique match ID from URL"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:10]
    
    def main(self, url: str, actual_result: Optional[str] = None):
        """Main execution function with learning integration"""
        
        print(f"\n{'🔍 ENHANCED FOOTBALL ANALYZER WITH MANUAL LEARNING 🔍'.center(70)}")
        print("─" * 70)
        print("📚 MANUAL LEARNING MODE: Use format 'H 3-2' to input results")
        print("─" * 70)
        
        print(f"\n📡 Fetching: {url[:70]}...")
        
        soup = self.fetch_page_soup(url)
        
        if not soup:
            print("❌ Failed to fetch page.")
            return
        
        # Extract basic information
        home_team, away_team, home_position, away_position = self.extract_team_names_and_positions(soup)
        
        if not home_team or not away_team:
            print("❌ Could not extract team names.")
            return
        
        print(f"✅ Match: {home_team} vs {away_team}")
        
        # Extract competition and match type
        competition = self.extract_competition_name(soup)
        match_type = self.determine_match_type(soup, competition)
        print(f"🏆 Competition: {competition}")
        
        # Extract H2H data
        h2h_matches = self.extract_head_to_head_matches(soup, home_team, away_team)
        
        if h2h_matches:
            print(f"🤝 Found {len(h2h_matches)} H2H matches")
        else:
            print("🤝 No H2H data")
        
        h2h_metrics = self.calculate_h2h_metrics(h2h_matches, home_team, away_team)
        
        # Extract recent matches with home/away separation
        html = str(soup)
        
        print(f"📈 Analyzing {home_team}...")
        home_matches = self.get_last_matches(html, home_team, competition, count=8)
        print(f"   Found {len(home_matches)} matches")
        
        print(f"📈 Analyzing {away_team}...")
        away_matches = self.get_last_matches(html, away_team, competition, count=8)
        print(f"   Found {len(away_matches)} matches")
        
        if len(home_matches) == 0 or len(away_matches) == 0:
            print("\n❌ Not enough match data.")
            return
        
        print(f"\n⚙️  Computing metrics...")
        home_metrics = self.decision_engine.compute_metrics_from_matches(
            home_team, home_matches, True, home_position, h2h_metrics, match_type
        )
        away_metrics = self.decision_engine.compute_metrics_from_matches(
            away_team, away_matches, False, away_position, H2HMetrics(), match_type
        )
        
        # Analyze core factors
        core_analysis = self.decision_engine.analyze_core_factors(home_metrics, away_metrics)
        
        # Get contextual factors based on match type
        match_summary_prelim = MatchSummary(is_derby=False)  # Simplified for factor calculation
        contextual_factors = self.adaptive_factors.get_contextual_factors(
            match_summary_prelim, home_metrics, away_metrics
        )
        
        # Apply learning adjustments to factors
        self.factors = self.decision_engine.learning_engine.adjust_factor_weights(contextual_factors)
        
        # Detect league for specific adjustments
        league = self.decision_engine.detect_league(competition, home_team, away_team)
        if league != competition:
            print(f"🏆 League: {league}")
        
        # Calculate probabilities with learning including league-specific adjustments
        probabilities = self.decision_engine.calculate_probabilities_with_learning(
            home_metrics, away_metrics, core_analysis, match_summary_prelim, self.factors, league
        )
        
        # Calculate goal markets
        goal_markets = self.calculate_goal_probabilities(home_metrics, away_metrics)
        
        # Analyze match summary
        match_summary = self.decision_engine.analyze_match_summary(
            probabilities, goal_markets, home_metrics, away_metrics, core_analysis
        )
        
        # Update factors with actual match context
        self.factors = self.adaptive_factors.get_contextual_factors(
            match_summary, home_metrics, away_metrics
        )
        
        # Make decisions
        decisions = self.decision_engine.make_decisions(
            probabilities, goal_markets, home_metrics, away_metrics, 
            core_analysis, match_summary, self.factors
        )
        
        print("\n" + "─" * 70)
        print(" 📊 ENHANCED ANALYSIS COMPLETE ".center(70, '═'))
        print("─" * 70 + "\n")
        
        # Display analysis
        self.display_analysis(
            home_team, away_team, home_metrics, away_metrics,
            probabilities, goal_markets, competition, home_matches, away_matches,
            decisions, core_analysis, match_summary, h2h_matches, self.factors, html
        )
        
        # Manual learning: If actual result provided, record for learning
        if actual_result:
            match_id = self.generate_match_id(url)
            primary_bet = decisions.get('primary_bet')
            
        # Record learning with goal outcomes
        if actual_result:
            # Parse goal outcomes if provided (e.g., "H 2-1")
            parts = actual_result.split()
            result_char = parts[0].upper()
            score = parts[1] if len(parts) > 1 else None
            
            # Process goal learning if score provided
            if score and '-' in score:
                try:
                    home_goals, away_goals = map(int, score.split('-'))
                    total_goals = home_goals + away_goals
                    
                    # Learn from goal predictions
                    predicted_over25 = goal_markets.get('over_25', 50) > 50
                    actual_over25 = total_goals > 2.5
                    
                    predicted_bts = goal_markets.get('bts', 50) > 50
                    actual_bts = home_goals > 0 and away_goals > 0
                    
                    # Record goal learning
                    self.decision_engine.learning_engine.record_goal_outcome(
                        predicted_over25, actual_over25,
                        predicted_bts, actual_bts,
                        total_goals, self.factors
                    )
                    
                    print(f"📚 Goal learning: O2.5 {'✅' if predicted_over25 == actual_over25 else '❌'} "
                          f"({total_goals} goals), BTS {'✅' if predicted_bts == actual_bts else '❌'}")
                except ValueError:
                    pass
            
            # Check if we have a primary bet prediction
            if primary_bet and primary_bet.get('selection'):
                predicted_result = {'HOME_WIN': 'H', 'AWAY_WIN': 'A', 'DRAW': 'D'}.get(primary_bet.get('selection', ''), 'U')
                
                if predicted_result != 'U':
                    self.decision_engine.record_match_outcome(
                        match_id, home_team, away_team, predicted_result,
                        probabilities, decisions.get('confidence_score', 0),
                        decisions.get('key_insights', []), decisions.get('match_characteristics', []),
                        home_metrics, away_metrics, core_analysis, actual_result, league
                    )
                    print(f"\n📚 Result recorded for learning: {actual_result}")
                else:
                    print(f"\n⚠️ Could not determine predicted result from primary bet")
            else:
                # Try to determine the highest probability outcome
                print(f"\n⚠️ No primary bet recommendation generated")
                
                # Use the highest probability as the prediction
                max_prob_key = max(probabilities.items(), key=lambda x: x[1])[0]
                predicted_result = {'home_win': 'H', 'away_win': 'A', 'draw': 'D'}.get(max_prob_key, 'U')
                
                if predicted_result != 'U':
                    print(f"📊 Using highest probability: {max_prob_key} ({probabilities[max_prob_key]:.1f}%)")
                    
                    self.decision_engine.record_match_outcome(
                        match_id, home_team, away_team, predicted_result,
                        probabilities, decisions.get('confidence_score', 0),
                        decisions.get('key_insights', []), decisions.get('match_characteristics', []),
                        home_metrics, away_metrics, core_analysis, actual_result, league
                    )
                    print(f"📚 Result recorded for learning: {actual_result}")
                else:
                    print(f"❌ Could not record result - no valid prediction")

# Main execution
if __name__ == "__main__":
    import sys
    
    # Initialize analyzer (no auto-verification)
    analyzer = EnhancedForebetAnalyzer()
    
    if len(sys.argv) < 2:
        print("Usage: python hw.py <forebet_url> [actual_result]")
        print("Example: python hw.py https://www.forebet.com/en/football/matches/team1-team2-12345")
        print("Manual Learning: Add H/A/D for actual result to record for learning")
        print("Example with result: python hw.py <url> \"H 3-2\"")
        sys.exit(1)
    
    url = sys.argv[1]
    actual_result = sys.argv[2] if len(sys.argv) > 2 else None
    
    if actual_result:
        # Allow formats: "H", "A", "D" or "H 2-1", "A 1-0", "D 1-1"
        parts = actual_result.split()
        result_char = parts[0].upper()
        if result_char not in ['H', 'A', 'D']:
            print(f"Warning: Invalid result type '{result_char}'. Must be H, A, or D.")
            actual_result = None
        elif len(parts) > 1:
            score = parts[1]
            if not re.match(r'^\d+-\d+$', score):
                print(f"Warning: Invalid score format '{score}'. Use format like '2-1'.")
                actual_result = result_char  # Keep just the result
    
    analyzer.main(url, actual_result)
