#!/usr/bin/env python3
"""
Merged Football Analyzer - Combines hw.py and predictor.py
Runs both analysis methods and displays comprehensive results
"""

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

# AI News Analysis
@dataclass
class NewsImpact:
    """News impact on team performance"""
    team_name: str
    injury_impact: float = 0.0  # -0.3 to 0.3
    lineup_strength: float = 0.0  # -0.2 to 0.2
    morale_factor: float = 0.0  # -0.2 to 0.2
    tactical_change: float = 0.0  # -0.1 to 0.1
    news_summary: List[str] = None
    confidence: float = 0.5  # 0-1
    
    def __post_init__(self):
        if self.news_summary is None:
            self.news_summary = []
    
    @property
    def total_impact(self) -> float:
        """Calculate total impact on team performance"""
        return self.injury_impact + self.lineup_strength + self.morale_factor + self.tactical_change

# ============================================================================
# SHARED DATA STRUCTURES
# ============================================================================

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

# ============================================================================
# HW.PY CLASSES (Enhanced Analysis)
# ============================================================================

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

# ============================================================================
# PREDICTOR.PY CLASSES (Simplified Analysis)
# ============================================================================

@dataclass
class TeamForm:
    """Simplified team form metrics"""
    team_name: str
    recent_form: float  # Quality-adjusted form from last 6 matches
    home_form: float    # Home-specific form
    away_form: float    # Away-specific form
    attack_strength: float
    defense_strength: float
    consistency: float
    league_position: Optional[int] = None

@dataclass
class MatchContext:
    """Match-specific context"""
    h2h_home_wins: int = 0
    h2h_away_wins: int = 0
    h2h_draws: int = 0
    h2h_total: int = 0
    home_advantage: float = 0.0
    quality_gap: float = 0.0

@dataclass
class Prediction:
    """Final prediction with confidence"""
    home_prob: float
    away_prob: float
    draw_prob: float
    recommended_bet: Optional[str] = None
    confidence: str = "LOW"
    reasoning: List[str] = None
    home_team: str = ""
    away_team: str = ""
    
    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = []

# ============================================================================
# SHARED DATA EXTRACTOR
# ============================================================================

class DataExtractor:
    """Shared data extraction functionality"""
    
    def __init__(self):
        self.team_cache = {}
        self.league_standings = {}
    
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
        
        return None
    
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
        
        return "Unknown Competition"

# ============================================================================
# ENHANCED ANALYZER (from hw.py)
# ============================================================================

class LearningEngine:
    """Analyze prediction outcomes and adjust weights"""
    
    def __init__(self, storage_file="prediction_history.json"):
        self.storage_file = storage_file
        self.history = self.load_history()
        self.learning_metrics = self.load_metrics()
    
    def load_history(self) -> List[PredictionOutcome]:
        """Load prediction history from file"""
        try:
            if Path(self.storage_file).exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                history = []
                for item in data:
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
        
        self.learning_metrics.accuracy = (
            self.learning_metrics.correct_predictions / 
            max(1, self.learning_metrics.total_predictions)
        )
        
        if self.learning_metrics.total_predictions % 10 == 0:
            self.save_history()
            self.save_metrics()

class EnhancedAnalyzer:
    """Enhanced analysis engine from hw.py"""
    
    def __init__(self):
        self.learning_engine = LearningEngine()
        self.factors = MatchFactors()
    
    def calculate_quality_adjusted_form(self, matches: List[Match]) -> float:
        """Calculate quality-adjusted form from last 6 matches"""
        if not matches:
            return 0.5
        
        form_matches = matches[:6] if len(matches) >= 6 else matches
        quality_adjusted_points = 0
        
        for match in form_matches:
            match_points = 3 if match.result == 'W' else 1 if match.result == 'D' else 0
            
            if match.opponent_position:
                if match.opponent_position <= 6:
                    quality_factor = 1.2 + (6 - match.opponent_position) * 0.05
                elif match.opponent_position <= 12:
                    quality_factor = 1.0
                else:
                    quality_factor = 0.9 - (match.opponent_position - 13) * 0.03
                
                quality_adjusted_points += match_points * quality_factor
            else:
                quality_adjusted_points += match_points
        
        max_points = len(form_matches) * 3
        return min(1.0, quality_adjusted_points / max_points) if max_points > 0 else 0.5
    
    def compute_enhanced_metrics(self, team_name: str, matches: List[Match], 
                                is_home_team: bool, league_position: Optional[int],
                                h2h_metrics: Optional[H2HMetrics] = None) -> TeamMetrics:
        """Compute comprehensive team metrics"""
        
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
        
        # Use last 6 matches for form calculation
        form_matches = matches[:6] if len(matches) >= 6 else matches
        total_matches = len(matches)
        recent_matches = min(3, len(form_matches))
        
        # Quality-adjusted form calculation
        form_rating = self.calculate_quality_adjusted_form(matches)
        
        # Recent form (last 3 matches)
        recent_form_matches = matches[:3] if len(matches) >= 3 else matches
        recent_form_rating = self.calculate_quality_adjusted_form(recent_form_matches)
        
        # Goals statistics
        goals_scored = [m.goals_scored for m in matches]
        goals_conceded = [m.goals_conceded for m in matches]
        
        goal_scoring_rate = sum(goals_scored) / total_matches if goals_scored else 1.0
        goal_conceding_rate = sum(goals_conceded) / total_matches if goals_conceded else 1.0
        
        # Home/away specific rates
        home_matches = [m for m in matches if m.venue == 'H']
        away_matches = [m for m in matches if m.venue == 'A']
        
        home_scoring_rate = statistics.mean([m.goals_scored for m in home_matches]) if home_matches else goal_scoring_rate
        away_scoring_rate = statistics.mean([m.goals_scored for m in away_matches]) if away_matches else goal_scoring_rate
        home_conceding_rate = statistics.mean([m.goals_conceded for m in home_matches]) if home_matches else goal_conceding_rate
        away_conceding_rate = statistics.mean([m.goals_conceded for m in away_matches]) if away_matches else goal_conceding_rate
        
        # Clean sheet rate
        clean_sheets = sum(1 for m in matches if m.goals_conceded == 0)
        clean_sheet_rate = clean_sheets / total_matches if total_matches > 0 else 0.3
        
        # Performance against quality opponents
        top6_opponents = 0
        points_vs_top6 = 0
        matches_vs_top6 = 0
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
        
        avg_opponent_position = statistics.mean(opponent_positions) if opponent_positions else 10.0
        performance_vs_top6 = points_vs_top6 / (max(1, matches_vs_top6) * 3)
        
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
        
        # Top6 performance score
        if matches_vs_top6 > 0:
            base_score = (points_vs_top6 / matches_vs_top6) / 3 * 10
            goals_scored_vs_top6 = sum(m.goals_scored for m in matches if m.opponent_position and m.opponent_position <= 6)
            goals_conceded_vs_top6 = sum(m.goals_conceded for m in matches if m.opponent_position and m.opponent_position <= 6)
            goal_diff_per_match = (goals_scored_vs_top6 - goals_conceded_vs_top6) / matches_vs_top6
            goal_diff_adjustment = goal_diff_per_match * 0.8 if goal_diff_per_match > 0 else goal_diff_per_match * 0.6
            top6_performance_score = max(0, min(10, base_score + goal_diff_adjustment))
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
        
        # Shots efficiency (estimated from goals)
        if goal_scoring_rate > 2.5:
            shots_efficiency = 0.30
        elif goal_scoring_rate > 2.0:
            shots_efficiency = 0.25
        elif goal_scoring_rate > 1.5:
            shots_efficiency = 0.20
        else:
            shots_efficiency = 0.15
        
        # Possession dominance (estimated)
        goal_diff_per_match = goal_scoring_rate - goal_conceding_rate
        if goal_diff_per_match > 1.5:
            possession_dominance = 0.80
        elif goal_diff_per_match > 1.0:
            possession_dominance = 0.70
        elif goal_diff_per_match > 0.5:
            possession_dominance = 0.60
        elif goal_diff_per_match > 0:
            possession_dominance = 0.55
        else:
            possession_dominance = 0.50
        
        # Home/away advantage calculation
        home_advantage = 0.0
        away_performance = 0.5
        
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
        
        # Momentum calculation
        momentum = 0.0
        recent_momentum = 0.0
        
        if len(matches) >= 6:
            last_3 = matches[:3]
            previous_3 = matches[3:6]
            
            last_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in last_3)
            previous_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in previous_3)
            
            momentum = (last_3_points - previous_3_points) / 9
            recent_momentum = (last_3_points / 9 - 0.5) * 2
        
        # Goal trends
        goal_trend = 0.0
        defense_trend = 0.0
        
        if len(matches) >= 6:
            last_3_goals = sum(m.goals_scored for m in matches[:3])
            previous_3_goals = sum(m.goals_scored for m in matches[3:6])
            goal_trend = max(-1.0, min(1.0, (last_3_goals - previous_3_goals) / 6.0))
            
            last_3_conceded = sum(m.goals_conceded for m in matches[:3])
            previous_3_conceded = sum(m.goals_conceded for m in matches[3:6])
            defense_trend = max(-1.0, min(1.0, (previous_3_conceded - last_3_conceded) / 6.0))
        
        # Consistency calculation
        match_results = [m.result for m in matches]
        if match_results:
            wins = match_results.count('W')
            draws = match_results.count('D')
            losses = match_results.count('L')
            total = len(match_results)
            
            win_rate = wins / total
            draw_rate = draws / total
            loss_rate = losses / total
            
            # Pattern-based consistency
            if win_rate >= 0.67:
                consistency_score = 0.8 + (win_rate * 0.2)
            elif loss_rate >= 0.67:
                consistency_score = 0.7 + (loss_rate * 0.2)
            elif draw_rate >= 0.5:
                consistency_score = 0.75 + (draw_rate * 0.15)
            else:
                dominant_rate = max(win_rate, draw_rate, loss_rate)
                consistency_score = 0.6 + (dominant_rate * 0.3)
            
            consistency_score = max(0.3, min(1.0, consistency_score))
        else:
            consistency_score = 0.5
        
        # Draw tendency
        draws = sum(1 for m in matches if m.result == 'D')
        draw_tendency = draws / total_matches if total_matches > 0 else 0.3
        
        # Big win/loss tendency
        big_wins = sum(1 for m in matches if m.result == 'W' and m.goals_scored - m.goals_conceded >= 2)
        big_losses = sum(1 for m in matches if m.result == 'L' and m.goals_conceded - m.goals_scored >= 2)
        
        big_win_tendency = big_wins / total_matches if total_matches > 0 else 0.1
        big_loss_tendency = big_losses / total_matches if total_matches > 0 else 0.1
        
        # Comeback ability
        wins = sum(1 for m in matches if m.result == 'W')
        comeback_ability = wins / max(1, total_matches - draws) * 0.7 if total_matches > draws else 0.5
        
        # Hold lead ability
        hold_lead_ability = 1 - big_loss_tendency
        
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
            vs_bottom_half_performance=0.5,
            home_scoring_rate=home_scoring_rate,
            away_scoring_rate=away_scoring_rate,
            home_conceding_rate=home_conceding_rate,
            away_conceding_rate=away_conceding_rate,
            home_away_trend=0.0,
            result_consistency=0.5,
            performance_variance=0.0,
            match_type_adjustment=1.0,
            league_tier_factor=1.0,
            position_trend=0,
            h2h_metrics=h2h_metrics or H2HMetrics(),
            late_goal_tendency=0.3,
            comeback_ability=comeback_ability,
            hold_lead_ability=hold_lead_ability,
            draw_tendency=draw_tendency,
            big_win_tendency=big_win_tendency,
            big_loss_tendency=big_loss_tendency,
            recent_fixture_difficulty=0.0,
            days_since_last_match=0,
            injury_impact_score=1.0,
            manager_stability=1.0,
            motivation_factor=1.0
        )
    
    def calculate_enhanced_probabilities(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics) -> Dict[str, float]:
        """Calculate probabilities using enhanced method"""
        
        home_score = 0
        away_score = 0
        
        # 1. Form rating (15%)
        home_score += home_metrics.form_rating * self.factors.form_weight
        away_score += away_metrics.form_rating * self.factors.form_weight
        
        # 2. Recent form (14%)
        home_score += home_metrics.recent_form_rating * self.factors.recent_form_weight
        away_score += away_metrics.recent_form_rating * self.factors.recent_form_weight
        
        # 3. Attack vs Defense (20%)
        home_attack_vs_defense = (home_metrics.home_scoring_rate * 0.6 + 
                                 (2 - away_metrics.away_conceding_rate) * 0.4) / 2
        away_attack_vs_defense = (away_metrics.away_scoring_rate * 0.6 + 
                                 (2 - home_metrics.home_conceding_rate) * 0.4) / 2
        
        home_score += home_attack_vs_defense * self.factors.attack_vs_defense_weight
        away_score += away_attack_vs_defense * self.factors.attack_vs_defense_weight
        
        # 4. Opponent quality (18%)
        home_opponent_quality = home_metrics.opponent_strength_factor * 0.6 + home_metrics.performance_vs_top6 * 0.4
        away_opponent_quality = away_metrics.opponent_strength_factor * 0.6 + away_metrics.performance_vs_top6 * 0.4
        
        home_score += home_opponent_quality * self.factors.opponent_quality_weight
        away_score += away_opponent_quality * self.factors.opponent_quality_weight
        
        # 5. Home advantage (12%)
        home_advantage_factor = (home_metrics.home_advantage + 1) / 2
        away_disadvantage_factor = (1 - home_metrics.home_advantage) / 2
        
        if away_metrics.away_performance > 0.7:
            home_advantage_factor *= 0.8
            away_disadvantage_factor *= 1.2
        
        home_score += home_advantage_factor * self.factors.home_advantage_weight
        away_score += away_disadvantage_factor * self.factors.home_advantage_weight
        
        # 6. Momentum (6%)
        home_momentum = (home_metrics.recent_momentum + 1) / 2
        away_momentum = (away_metrics.recent_momentum + 1) / 2
        
        home_score += home_momentum * self.factors.momentum_weight
        away_score += away_momentum * self.factors.momentum_weight
        
        # 7. Consistency (10%)
        home_score += home_metrics.consistency_score * self.factors.consistency_weight
        away_score += away_metrics.consistency_score * self.factors.consistency_weight
        
        # 8. H2H factor (16%)
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
        
        # Base draw probability
        closeness = 1 - abs(home_score - away_score)
        base_draw = 0.33 * closeness
        
        # Adjust for draw tendencies
        draw_tendency_factor = (home_metrics.draw_tendency + away_metrics.draw_tendency) / 2
        if draw_tendency_factor > 0.4:
            base_draw *= 1.3
        
        # Normalize
        total = home_score + away_score + base_draw
        home_prob = home_score / total * 100
        away_prob = away_score / total * 100
        draw_prob = base_draw / total * 100
        
        return {
            'home_win': home_prob,
            'away_win': away_prob,
            'draw': draw_prob
        }

# ============================================================================
# SIMPLIFIED PREDICTOR (from predictor.py)
# ============================================================================

class SimplifiedPredictor:
    """Simplified prediction model with best practices"""
    
    def calculate_quality_adjusted_form(self, matches: List[Match]) -> float:
        """Calculate quality-adjusted form - best practice from hw.py"""
        if not matches:
            return 0.5
        
        quality_points = 0
        max_points = 0
        
        for match in matches:
            match_points = 3 if match.result == 'W' else 1 if match.result == 'D' else 0
            
            # Quality adjustment based on opponent strength
            if match.opponent_position:
                if match.opponent_position <= 6:
                    quality_factor = 1.2 + (6 - match.opponent_position) * 0.05
                elif match.opponent_position <= 12:
                    quality_factor = 1.0
                else:
                    quality_factor = 0.9 - (match.opponent_position - 13) * 0.03
                
                quality_points += match_points * quality_factor
            else:
                quality_points += match_points
            
            max_points += 3
        
        return min(1.0, quality_points / max_points) if max_points > 0 else 0.5
    
    def calculate_venue_form(self, matches: List[Match], venue: str) -> float:
        """Calculate venue-specific form"""
        venue_matches = [m for m in matches if m.venue == venue]
        if not venue_matches:
            return 0.5
        
        points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in venue_matches)
        max_points = len(venue_matches) * 3
        
        return points / max_points if max_points > 0 else 0.5
    
    def calculate_attack_defense_strength(self, matches: List[Match]) -> Tuple[float, float]:
        """Calculate attack and defense strength"""
        if not matches:
            return 1.0, 1.0
        
        goals_scored = [m.goals_scored for m in matches]
        goals_conceded = [m.goals_conceded for m in matches]
        
        attack_strength = statistics.mean(goals_scored) if goals_scored else 1.0
        defense_strength = 2.0 - statistics.mean(goals_conceded) if goals_conceded else 1.0  # Inverted
        
        return attack_strength, max(0.1, defense_strength)
    
    def calculate_consistency(self, matches: List[Match]) -> float:
        """Calculate team consistency"""
        if len(matches) < 3:
            return 0.5
        
        results = [m.result for m in matches]
        wins = results.count('W')
        draws = results.count('D')
        losses = results.count('L')
        total = len(results)
        
        # High consistency for teams with clear patterns
        dominant_result = max(wins, draws, losses)
        consistency = (dominant_result / total) * 0.8 + 0.2
        
        return min(1.0, consistency)
    
    def compute_team_form(self, team_name: str, matches: List[Match], position: Optional[int]) -> TeamForm:
        """Compute comprehensive team metrics"""
        recent_form = self.calculate_quality_adjusted_form(matches)
        home_form = self.calculate_venue_form(matches, 'H')
        away_form = self.calculate_venue_form(matches, 'A')
        attack_strength, defense_strength = self.calculate_attack_defense_strength(matches)
        consistency = self.calculate_consistency(matches)
        
        return TeamForm(
            team_name=team_name,
            recent_form=recent_form,
            home_form=home_form,
            away_form=away_form,
            attack_strength=attack_strength,
            defense_strength=defense_strength,
            consistency=consistency,
            league_position=position
        )
    
    def calculate_simplified_probabilities(self, home_form: TeamForm, away_form: TeamForm, context: MatchContext) -> Prediction:
        """Calculate match probabilities using best practices"""
        
        # Core factors with optimized weights
        home_score = 0
        away_score = 0
        
        # 1. Recent form (25% weight)
        home_score += home_form.recent_form * 0.25
        away_score += away_form.recent_form * 0.25
        
        # 2. Venue-specific form (20% weight)
        home_score += home_form.home_form * 0.20
        away_score += away_form.away_form * 0.20
        
        # 3. Attack vs Defense matchup (25% weight)
        home_attack_advantage = (home_form.attack_strength / max(0.1, away_form.defense_strength)) * 0.125
        away_attack_advantage = (away_form.attack_strength / max(0.1, home_form.defense_strength)) * 0.125
        
        home_score += home_attack_advantage
        away_score += away_attack_advantage
        
        # 4. Consistency factor (15% weight)
        home_score += home_form.consistency * 0.15
        away_score += away_form.consistency * 0.15
        
        # 5. League position factor (10% weight)
        if home_form.league_position and away_form.league_position:
            position_advantage = (away_form.league_position - home_form.league_position) / 20.0
            home_score += (0.5 + position_advantage) * 0.10
            away_score += (0.5 - position_advantage) * 0.10
        else:
            home_score += 0.05
            away_score += 0.05
        
        # 6. H2H factor (5% weight)
        if context.h2h_total >= 3:
            h2h_home_rate = context.h2h_home_wins / context.h2h_total
            h2h_away_rate = context.h2h_away_wins / context.h2h_total
            
            home_score += h2h_home_rate * 0.05
            away_score += h2h_away_rate * 0.05
        else:
            home_score += 0.025
            away_score += 0.025
        
        # Calculate draw probability
        form_difference = abs(home_form.recent_form - away_form.recent_form)
        base_draw = 0.30 - (form_difference * 0.2)  # Less draw when big form difference
        
        if context.h2h_total >= 3:
            h2h_draw_rate = context.h2h_draws / context.h2h_total
            if h2h_draw_rate > 0.4:
                base_draw *= 1.3  # Boost draw if H2H shows draw tendency
        
        # Normalize probabilities
        total = home_score + away_score + base_draw
        home_prob = (home_score / total) * 100
        away_prob = (away_score / total) * 100
        draw_prob = (base_draw / total) * 100
        
        # Determine recommendation and confidence
        max_prob = max(home_prob, away_prob, draw_prob)
        prob_spread = max_prob - min(home_prob, away_prob, draw_prob)
        
        recommended_bet = None
        confidence = "LOW"
        reasoning = []
        
        # Enhanced betting logic - always recommend something
        if prob_spread < 15:  # Teams are close - recommend draw
            recommended_bet = "DRAW"
            confidence = "MEDIUM" if prob_spread < 10 else "LOW"
            reasoning.append(f"Teams evenly matched - draw likely ({draw_prob:.1f}%)")
        elif max_prob > 45:  # Clear favorite
            if home_prob == max_prob:
                recommended_bet = "HOME_WIN"
                reasoning.append(f"Home team superior ({home_prob:.1f}%)")
            elif away_prob == max_prob:
                recommended_bet = "AWAY_WIN"
                reasoning.append(f"Away team superior ({away_prob:.1f}%)")
            else:
                recommended_bet = "DRAW"
                reasoning.append(f"Draw most likely ({draw_prob:.1f}%)")
            
            confidence = "HIGH" if max_prob > 55 else "MEDIUM"
        else:  # Moderate favorite
            if home_prob > away_prob and home_prob > draw_prob:
                recommended_bet = "HOME_WIN"
                reasoning.append(f"Home team favored ({home_prob:.1f}%)")
            elif away_prob > home_prob and away_prob > draw_prob:
                recommended_bet = "AWAY_WIN"
                reasoning.append(f"Away team favored ({away_prob:.1f}%)")
            else:
                recommended_bet = "DRAW"
                reasoning.append(f"Draw most likely ({draw_prob:.1f}%)")
            
            confidence = "MEDIUM" if max_prob > 40 else "LOW"
        
        # Add reasoning based on key factors
        if abs(home_form.recent_form - away_form.recent_form) > 0.3:
            better_form = "Home" if home_form.recent_form > away_form.recent_form else "Away"
            reasoning.append(f"{better_form} team has significantly better recent form")
        
        if home_form.home_form > 0.7:
            reasoning.append("Home team very strong at home")
        elif away_form.away_form > 0.7:
            reasoning.append("Away team excellent on the road")
        
        return Prediction(
            home_prob=home_prob,
            away_prob=away_prob,
            draw_prob=draw_prob,
            recommended_bet=recommended_bet,
            confidence=confidence,
            reasoning=reasoning,
            home_team=home_form.team_name,
            away_team=away_form.team_name
        )

# ============================================================================
# MERGED ANALYZER - MAIN CLASS
# ============================================================================

class MergedFootballAnalyzer:
    """Merged analyzer that runs both hw.py and predictor.py methods"""
    
    def __init__(self):
        self.extractor = DataExtractor()
        self.enhanced_analyzer = EnhancedAnalyzer()
        self.simplified_predictor = SimplifiedPredictor()
    
    def extract_h2h_data(self, soup: BeautifulSoup, home_team: str, away_team: str) -> Tuple[H2HMetrics, MatchContext]:
        """Extract head-to-head data for both analyzers"""
        h2h_metrics = H2HMetrics()
        match_context = MatchContext()
        
        # Look for H2H sections
        h2h_text = str(soup)
        
        # Simple H2H extraction
        h2h_patterns = [
            r'(\d+)\s*[–\-]\s*(\d+)',
        ]
        
        matches_found = 0
        for pattern in h2h_patterns:
            for match in re.finditer(pattern, h2h_text):
                try:
                    score1 = int(match.group(1))
                    score2 = int(match.group(2))
                    
                    h2h_metrics.total_matches += 1
                    match_context.h2h_total += 1
                    
                    if score1 > score2:
                        h2h_metrics.home_wins += 1
                        match_context.h2h_home_wins += 1
                    elif score2 > score1:
                        h2h_metrics.away_wins += 1
                        match_context.h2h_away_wins += 1
                    else:
                        h2h_metrics.draws += 1
                        match_context.h2h_draws += 1
                    
                    matches_found += 1
                    if matches_found >= 10:  # Limit to avoid noise
                        break
                except:
                    continue
        
        # Calculate rates for H2HMetrics
        if h2h_metrics.total_matches > 0:
            h2h_metrics.home_win_rate = h2h_metrics.home_wins / h2h_metrics.total_matches
            h2h_metrics.away_win_rate = h2h_metrics.away_wins / h2h_metrics.total_matches
            h2h_metrics.draw_rate = h2h_metrics.draws / h2h_metrics.total_matches
        
        return h2h_metrics, match_context
    
    def calculate_goal_probabilities(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics) -> Dict[str, float]:
        """Calculate goal market probabilities"""
        
        # Expected goals calculation
        home_expected = (home_metrics.home_scoring_rate * 0.7 + 
                        (2 - away_metrics.away_conceding_rate) * 0.3)
        away_expected = (away_metrics.away_scoring_rate * 0.7 + 
                        (2 - home_metrics.home_conceding_rate) * 0.3)
        
        # Adjust for recent form
        home_expected *= (0.8 + home_metrics.recent_form_rating * 0.4)
        away_expected *= (0.8 + away_metrics.recent_form_rating * 0.4)
        
        # Bounds
        home_expected = max(0.1, min(4.0, home_expected))
        away_expected = max(0.1, min(3.5, away_expected))
        
        total_expected = home_expected + away_expected
        
        # Calculate probabilities using Poisson distribution approximation
        def poisson_prob(lambda_, k):
            if k > 8: return 0
            return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)
        
        over_15_prob = 0
        over_25_prob = 0
        over_35_prob = 0
        bts_prob = 0
        
        for i in range(9):
            for j in range(9):
                prob = poisson_prob(home_expected, i) * poisson_prob(away_expected, j)
                total_goals = i + j
                
                if total_goals > 1.5: over_15_prob += prob
                if total_goals > 2.5: over_25_prob += prob
                if total_goals > 3.5: over_35_prob += prob
                if i > 0 and j > 0: bts_prob += prob
        
        return {
            'over_15': over_15_prob * 100,
            'over_25': over_25_prob * 100,
            'over_35': over_35_prob * 100,
            'under_15': (1 - over_15_prob) * 100,
            'under_25': (1 - over_25_prob) * 100,
            'under_35': (1 - over_35_prob) * 100,
            'bts_yes': bts_prob * 100,
            'bts_no': (1 - bts_prob) * 100,
            'total_expected': total_expected
        }
    
    def get_team_league_position(self, team_name: str) -> str:
        """Get team's league position with better detection"""
        
        # First check league standings
        if hasattr(self, 'league_standings') and self.league_standings:
            team_norm = self.normalize_name(team_name)
            for team, pos in self.league_standings.items():
                if self.normalize_name(team) == team_norm:
                    return f"#{pos}"
        
        # Check extractor's league standings
        if hasattr(self.extractor, 'league_standings') and self.extractor.league_standings:
            team_norm = self.extractor.normalize_name(team_name)
            for team, pos in self.extractor.league_standings.items():
                if self.extractor.normalize_name(team) == team_norm:
                    return f"#{pos}"
        
        return "Position Unknown"
    
    def analyze_match(self, url: str, actual_result: Optional[str] = None):
        """Main analysis function that runs both methods"""
        
        print(f"\n{'🔍 MERGED FOOTBALL ANALYZER 🔍'.center(80)}")
        print("─" * 80)
        print("🎯 Running BOTH Enhanced Analysis (hw.py) AND Simplified Predictor (predictor.py)")
        print("─" * 80)
        
        print(f"\n📡 Fetching: {url[:70]}...")
        
        # Fetch and parse page
        soup = self.extractor.fetch_page_soup(url)
        if not soup:
            print("❌ Failed to fetch page")
            return
        
        # Extract basic info
        home_team, away_team, home_pos, away_pos = self.extractor.extract_team_names_and_positions(soup)
        if not home_team or not away_team:
            print("❌ Could not extract team names")
            return
        
        print(f"⚽ Match: {home_team} vs {away_team}")
        
        # Extract competition
        competition = self.extractor.extract_competition_name(soup)
        print(f"🏆 Competition: {competition}")
        
        # Extract match data
        html = str(soup)
        home_matches = self.extractor.get_last_matches(html, home_team, competition, 8)
        away_matches = self.extractor.get_last_matches(html, away_team, competition, 8)
        h2h_metrics, match_context = self.extract_h2h_data(soup, home_team, away_team)
        
        print(f"📊 Data: {len(home_matches)} home matches, {len(away_matches)} away matches")
        if h2h_metrics.total_matches > 0:
            print(f"🤝 H2H: {h2h_metrics.total_matches} matches found")
        
        # AI News Analysis
        print(f"🤖 Analyzing team news...")
        home_news = self.analyze_team_news(home_team, competition)
        away_news = self.analyze_team_news(away_team, competition)
        
        if not home_matches or not away_matches:
            print("❌ Insufficient match data")
            return
        
        print(f"\n{'═' * 80}")
        print(f"🔬 ENHANCED ANALYSIS (hw.py method)")
        print(f"{'═' * 80}")
        
        # Run enhanced analysis
        enhanced_home_metrics = self.enhanced_analyzer.compute_enhanced_metrics(
            home_team, home_matches, True, home_pos, h2h_metrics
        )
        enhanced_away_metrics = self.enhanced_analyzer.compute_enhanced_metrics(
            away_team, away_matches, False, away_pos, H2HMetrics()
        )
        
        enhanced_probabilities = self.enhanced_analyzer.calculate_enhanced_probabilities(
            enhanced_home_metrics, enhanced_away_metrics
        )
        
        # Apply news impact to probabilities
        enhanced_probabilities = self.apply_news_impact(enhanced_probabilities, home_news, away_news)
        
        self.display_enhanced_results(home_team, away_team, enhanced_home_metrics, 
                                    enhanced_away_metrics, enhanced_probabilities)
        
        # Display news analysis
        self.display_news_analysis(home_news, away_news)
        
        print(f"\n{'═' * 80}")
        print(f"🎯 SIMPLIFIED PREDICTOR (predictor.py method)")
        print(f"{'═' * 80}")
        
        # Run simplified analysis
        simplified_home_form = self.simplified_predictor.compute_team_form(home_team, home_matches, home_pos)
        simplified_away_form = self.simplified_predictor.compute_team_form(away_team, away_matches, away_pos)
        
        simplified_prediction = self.simplified_predictor.calculate_simplified_probabilities(
            simplified_home_form, simplified_away_form, match_context
        )
        
        self.display_simplified_results(simplified_home_form, simplified_away_form, simplified_prediction)
        
        print(f"\n{'═' * 80}")
        print(f"📊 COMPARISON & CONSENSUS")
        print(f"{'═' * 80}")
        
        self.display_comparison(enhanced_probabilities, simplified_prediction, home_team, away_team)
        
        # Add detailed last 6 analysis after comparison
        self.display_last_6_analysis(home_team, away_team, home_matches, away_matches)
        
        # Calculate and display goal predictions
        goal_probs = self.calculate_goal_probabilities(enhanced_home_metrics, enhanced_away_metrics)
        self.display_goal_predictions(goal_probs)
        
        # Handle learning if actual result provided
        if actual_result:
            self.record_learning(actual_result, enhanced_probabilities, simplified_prediction, 
                               home_team, away_team, url, goal_probs)
        
        print(f"\n{'─' * 80}")
        print(f"📚 Manual Learning: Add result with 'H 2-1', 'A 1-0', or 'D 1-1'")
        print(f"{'─' * 80}")
    
    def display_enhanced_results(self, home_team: str, away_team: str, 
                               home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                               probabilities: Dict[str, float]):
        """Display enhanced analysis results"""
        
        home_display = self.extractor.clean_team_name_display(home_team)
        away_display = self.extractor.clean_team_name_display(away_team)
        
        print(f"\n📊 ENHANCED TEAM METRICS")
        print(f"{'Metric':<25} {'Home':<15} {'Away':<15}")
        print("-" * 55)
        print(f"Form Rating              {home_metrics.form_rating:.1%}           {away_metrics.form_rating:.1%}")
        print(f"Recent Form              {home_metrics.recent_form_rating:.1%}           {away_metrics.recent_form_rating:.1%}")
        print(f"Goals/Match              {home_metrics.goal_scoring_rate:.1f}             {away_metrics.goal_scoring_rate:.1f}")
        print(f"Home/Away Performance    {home_metrics.home_advantage:.1%}           {away_metrics.away_performance:.1%}")
        print(f"Consistency              {home_metrics.consistency_score:.1%}           {away_metrics.consistency_score:.1%}")
        print(f"vs Top6 Performance      {home_metrics.performance_vs_top6:.1%}           {away_metrics.performance_vs_top6:.1%}")
        
        print(f"\n📈 ENHANCED PROBABILITIES")
        print(f"{home_display[:15]:<15} {probabilities['home_win']:5.1f}%")
        print(f"{'Draw':<15} {probabilities['draw']:5.1f}%")
        print(f"{away_display[:15]:<15} {probabilities['away_win']:5.1f}%")
        
        # Determine enhanced recommendation
        max_prob = max(probabilities.values())
        if probabilities['home_win'] == max_prob:
            enhanced_rec = f"HOME WIN ({probabilities['home_win']:.1f}%)"
        elif probabilities['away_win'] == max_prob:
            enhanced_rec = f"AWAY WIN ({probabilities['away_win']:.1f}%)"
        else:
            enhanced_rec = f"DRAW ({probabilities['draw']:.1f}%)"
        
        print(f"\n🎯 Enhanced Recommendation: {enhanced_rec}")
    
    def display_last_6_analysis(self, home_team: str, away_team: str, 
                               home_matches: List[Match], away_matches: List[Match]):
        """Display detailed last 6 matches analysis"""
        
        home_display = self.extractor.clean_team_name_display(home_team)
        away_display = self.extractor.clean_team_name_display(away_team)
        
        print(f"\n{'📋 DETAILED LAST 6 MATCHES ANALYSIS'.center(80, '═')}")
        
        # Home team analysis
        home_league_pos = self.get_team_league_position(home_team)
        
        print(f"\n🏠 {home_display} ({home_league_pos})")
        print("─" * 50)
        
        if home_matches:
            home_results = []
            home_goals_for = 0
            home_goals_against = 0
            home_points = 0
            
            for i, match in enumerate(home_matches[:6], 1):
                result_icon = {'W': '✅', 'D': '⚪', 'L': '❌'}[match.result]
                venue_icon = '🏠' if match.venue == 'H' else '🛣️'
                
                # Get opponent info
                opponent = match.away_team if match.venue == 'H' else match.home_team
                opponent_clean = self.extractor.clean_team_name_display(opponent)
                opp_pos = f"#{match.opponent_position}" if match.opponent_position else "N/A"
                
                print(f"{i}. {result_icon} {venue_icon} {match.score} vs {opponent_clean[:20]} ({opp_pos})")
                
                home_goals_for += match.goals_scored
                home_goals_against += match.goals_conceded
                home_points += 3 if match.result == 'W' else 1 if match.result == 'D' else 0
            
            # Summary stats
            matches_count = min(6, len(home_matches))
            avg_goals_for = home_goals_for / matches_count
            avg_goals_against = home_goals_against / matches_count
            points_per_game = home_points / matches_count
            
            print(f"\n📊 Summary: {home_points} pts ({points_per_game:.1f} ppg) | "
                  f"Goals: {avg_goals_for:.1f} for, {avg_goals_against:.1f} against | "
                  f"Form: {home_points/18:.1%}")
        else:
            print("No recent matches found")
        
        # Away team analysis
        away_league_pos = self.get_team_league_position(away_team)
        
        print(f"\n🛣️ {away_display} ({away_league_pos})")
        print("─" * 50)
        
        if away_matches:
            away_results = []
            away_goals_for = 0
            away_goals_against = 0
            away_points = 0
            
            for i, match in enumerate(away_matches[:6], 1):
                result_icon = {'W': '✅', 'D': '⚪', 'L': '❌'}[match.result]
                venue_icon = '🏠' if match.venue == 'H' else '🛣️'
                
                # Get opponent info
                opponent = match.away_team if match.venue == 'H' else match.home_team
                opponent_clean = self.extractor.clean_team_name_display(opponent)
                opp_pos = f"#{match.opponent_position}" if match.opponent_position else "N/A"
                
                print(f"{i}. {result_icon} {venue_icon} {match.score} vs {opponent_clean[:20]} ({opp_pos})")
                
                away_goals_for += match.goals_scored
                away_goals_against += match.goals_conceded
                away_points += 3 if match.result == 'W' else 1 if match.result == 'D' else 0
            
            # Summary stats
            matches_count = min(6, len(away_matches))
            avg_goals_for = away_goals_for / matches_count
            avg_goals_against = away_goals_against / matches_count
            points_per_game = away_points / matches_count
            
            print(f"\n📊 Summary: {away_points} pts ({points_per_game:.1f} ppg) | "
                  f"Goals: {avg_goals_for:.1f} for, {avg_goals_against:.1f} against | "
                  f"Form: {away_points/18:.1%}")
        else:
            print("No recent matches found")
        
        print(f"\n{'═' * 80}")
    
    def display_goal_predictions(self, goal_probs: Dict[str, float]):
        """Display over/under and goal market predictions"""
        
        print(f"\n{'⚽ GOAL MARKET PREDICTIONS'.center(80, '═')}")
        
        print(f"\n📊 TOTAL GOALS MARKETS")
        print(f"Expected Total Goals: {goal_probs['total_expected']:.1f}")
        print()
        
        # Over/Under markets with recommendations
        markets = [
            ('Over 1.5', goal_probs['over_15'], 'Under 1.5', goal_probs['under_15']),
            ('Over 2.5', goal_probs['over_25'], 'Under 2.5', goal_probs['under_25']),
            ('Over 3.5', goal_probs['over_35'], 'Under 3.5', goal_probs['under_35'])
        ]
        
        print(f"{'Market':<12} {'Probability':<12} {'Recommendation'}")
        print("-" * 50)
        
        for over_name, over_prob, under_name, under_prob in markets:
            if over_prob > under_prob:
                rec = f"✅ {over_name} ({over_prob:.1f}%)"
                confidence = "HIGH" if over_prob > 65 else "MEDIUM" if over_prob > 55 else "LOW"
            else:
                rec = f"✅ {under_name} ({under_prob:.1f}%)"
                confidence = "HIGH" if under_prob > 65 else "MEDIUM" if under_prob > 55 else "LOW"
            
            print(f"{over_name:<12} {over_prob:5.1f}%      {rec} ({confidence})")
        
        # Both Teams to Score
        print(f"\n📊 BOTH TEAMS TO SCORE")
        if goal_probs['bts_yes'] > goal_probs['bts_no']:
            bts_rec = f"✅ YES ({goal_probs['bts_yes']:.1f}%)"
            bts_conf = "HIGH" if goal_probs['bts_yes'] > 65 else "MEDIUM" if goal_probs['bts_yes'] > 55 else "LOW"
        else:
            bts_rec = f"✅ NO ({goal_probs['bts_no']:.1f}%)"
            bts_conf = "HIGH" if goal_probs['bts_no'] > 65 else "MEDIUM" if goal_probs['bts_no'] > 55 else "LOW"
        
        print(f"BTS Yes: {goal_probs['bts_yes']:5.1f}% | BTS No: {goal_probs['bts_no']:5.1f}%")
        print(f"Recommendation: {bts_rec} ({bts_conf})")
        
        print(f"\n{'═' * 80}")
    
    def apply_news_impact(self, probabilities: Dict[str, float], home_news: NewsImpact, away_news: NewsImpact) -> Dict[str, float]:
        """Apply news impact to match probabilities"""
        
        # Calculate net impact difference
        home_impact = home_news.total_impact
        away_impact = away_news.total_impact
        impact_diff = home_impact - away_impact
        
        # Apply impact with confidence weighting
        confidence_weight = (home_news.confidence + away_news.confidence) / 2
        adjusted_impact = impact_diff * confidence_weight * 10  # Scale to percentage points
        
        # Adjust probabilities
        new_probs = probabilities.copy()
        
        if adjusted_impact > 0:  # Favors home team
            new_probs['home_win'] += abs(adjusted_impact)
            new_probs['away_win'] -= abs(adjusted_impact) * 0.7
            new_probs['draw'] -= abs(adjusted_impact) * 0.3
        elif adjusted_impact < 0:  # Favors away team
            new_probs['away_win'] += abs(adjusted_impact)
            new_probs['home_win'] -= abs(adjusted_impact) * 0.7
            new_probs['draw'] -= abs(adjusted_impact) * 0.3
        
        # Ensure probabilities stay within bounds and sum to 100
        for key in new_probs:
            new_probs[key] = max(5, min(85, new_probs[key]))
        
        total = sum(new_probs.values())
        for key in new_probs:
            new_probs[key] = (new_probs[key] / total) * 100
        
        return new_probs
    
    def display_news_analysis(self, home_news: NewsImpact, away_news: NewsImpact):
        """Display AI news analysis results"""
        
        print(f"\n{'🤖 AI NEWS ANALYSIS'.center(80, '═')}")
        
        home_display = self.extractor.clean_team_name_display(home_news.team_name)
        away_display = self.extractor.clean_team_name_display(away_news.team_name)
        
        print(f"\n🏠 {home_display} News Impact: {home_news.total_impact:+.2f}")
        print(f"   Confidence: {home_news.confidence:.1%}")
        if home_news.news_summary:
            for summary in home_news.news_summary:
                print(f"   {summary}")
        else:
            print("   📰 No significant news found")
        
        print(f"\n🛣️ {away_display} News Impact: {away_news.total_impact:+.2f}")
        print(f"   Confidence: {away_news.confidence:.1%}")
        if away_news.news_summary:
            for summary in away_news.news_summary:
                print(f"   {summary}")
        else:
            print("   📰 No significant news found")
        
        # Overall impact assessment
        net_impact = home_news.total_impact - away_news.total_impact
        if abs(net_impact) > 0.1:
            if net_impact > 0:
                print(f"\n📈 NEWS ADVANTAGE: {home_display} (+{net_impact:.2f})")
            else:
                print(f"\n📈 NEWS ADVANTAGE: {away_display} (+{abs(net_impact):.2f})")
        else:
            print(f"\n⚖️ NEWS IMPACT: Neutral (minimal difference)")
        
        print(f"\n{'═' * 80}")
    
    def display_simplified_results(self, home_form: TeamForm, away_form: TeamForm, 
                                 prediction: Prediction):
        """Display simplified predictor results"""
        
        print(f"\n📊 SIMPLIFIED TEAM METRICS")
        print(f"{'Metric':<20} {'Home':<12} {'Away':<12}")
        print("-" * 45)
        print(f"Recent Form          {home_form.recent_form:.1%}        {away_form.recent_form:.1%}")
        print(f"Home/Away Form       {home_form.home_form:.1%}        {away_form.away_form:.1%}")
        print(f"Attack Strength      {home_form.attack_strength:.1f}          {away_form.attack_strength:.1f}")
        print(f"Defense Strength     {home_form.defense_strength:.1f}          {away_form.defense_strength:.1f}")
        print(f"Consistency          {home_form.consistency:.1%}        {away_form.consistency:.1%}")
        
        print(f"\n📈 SIMPLIFIED PROBABILITIES")
        print(f"{prediction.home_team[:15]:<15} {prediction.home_prob:5.1f}%")
        print(f"{'Draw':<15} {prediction.draw_prob:5.1f}%")
        print(f"{prediction.away_team[:15]:<15} {prediction.away_prob:5.1f}%")
        
        print(f"\n🎯 Simplified Recommendation: {prediction.recommended_bet} ({prediction.confidence})")
        
        if prediction.reasoning:
            print(f"💡 Reasoning:")
            for reason in prediction.reasoning[:3]:
                print(f"   • {reason}")
    
    def display_comparison(self, enhanced_probs: Dict[str, float], simplified_pred: Prediction,
                         home_team: str, away_team: str):
        """Display comparison between both methods"""
        
        home_display = self.extractor.clean_team_name_display(home_team)
        away_display = self.extractor.clean_team_name_display(away_team)
        
        print(f"\n📊 PROBABILITY COMPARISON")
        print(f"{'Outcome':<15} {'Enhanced':<12} {'Simplified':<12} {'Difference':<12}")
        print("-" * 52)
        
        home_diff = abs(enhanced_probs['home_win'] - simplified_pred.home_prob)
        away_diff = abs(enhanced_probs['away_win'] - simplified_pred.away_prob)
        draw_diff = abs(enhanced_probs['draw'] - simplified_pred.draw_prob)
        
        print(f"{home_display[:14]:<15} {enhanced_probs['home_win']:5.1f}%      {simplified_pred.home_prob:5.1f}%      {home_diff:5.1f}%")
        print(f"{'Draw':<15} {enhanced_probs['draw']:5.1f}%      {simplified_pred.draw_prob:5.1f}%      {draw_diff:5.1f}%")
        print(f"{away_display[:14]:<15} {enhanced_probs['away_win']:5.1f}%      {simplified_pred.away_prob:5.1f}%      {away_diff:5.1f}%")
        
        # Determine consensus
        enhanced_winner = max(enhanced_probs.items(), key=lambda x: x[1])
        simplified_winner_map = {
            'HOME_WIN': ('home_win', simplified_pred.home_prob),
            'AWAY_WIN': ('away_win', simplified_pred.away_prob),
            'DRAW': ('draw', simplified_pred.draw_prob)
        }
        simplified_winner = simplified_winner_map.get(simplified_pred.recommended_bet, ('draw', 0))
        
        print(f"\n🤝 CONSENSUS ANALYSIS")
        if enhanced_winner[0] == simplified_winner[0]:
            outcome_names = {'home_win': f'{home_display} Win', 'away_win': f'{away_display} Win', 'draw': 'Draw'}
            print(f"✅ BOTH METHODS AGREE: {outcome_names[enhanced_winner[0]]}")
            avg_prob = (enhanced_winner[1] + simplified_winner[1]) / 2
            print(f"📊 Average Probability: {avg_prob:.1f}%")
            print(f"🎯 CONSENSUS RECOMMENDATION: {outcome_names[enhanced_winner[0]]}")
            
            # Confidence based on agreement
            prob_diff = abs(enhanced_winner[1] - simplified_winner[1])
            if prob_diff < 5:
                consensus_confidence = "HIGH"
            elif prob_diff < 10:
                consensus_confidence = "MEDIUM"
            else:
                consensus_confidence = "LOW"
            print(f"📈 Consensus Confidence: {consensus_confidence}")
        else:
            print(f"❌ METHODS DISAGREE")
            outcome_names = {'home_win': f'{home_display} Win', 'away_win': f'{away_display} Win', 'draw': 'Draw'}
            print(f"   Enhanced: {outcome_names[enhanced_winner[0]]} ({enhanced_winner[1]:.1f}%)")
            print(f"   Simplified: {outcome_names[simplified_winner[0]]} ({simplified_winner[1]:.1f}%)")
            print(f"🎯 RECOMMENDATION: Consider both predictions - NO CLEAR CONSENSUS")
            print(f"📈 Consensus Confidence: VERY LOW")
        
        # Average probabilities for reference
        avg_home = (enhanced_probs['home_win'] + simplified_pred.home_prob) / 2
        avg_away = (enhanced_probs['away_win'] + simplified_pred.away_prob) / 2
        avg_draw = (enhanced_probs['draw'] + simplified_pred.draw_prob) / 2
        
        print(f"\n📊 AVERAGE PROBABILITIES")
        print(f"{home_display[:15]:<15} {avg_home:5.1f}%")
        print(f"{'Draw':<15} {avg_draw:5.1f}%")
        print(f"{away_display[:15]:<15} {avg_away:5.1f}%")
    
    def record_learning(self, actual_result: str, enhanced_probs: Dict[str, float], 
                       simplified_pred: Prediction, home_team: str, away_team: str, url: str,
                       goal_probs: Dict[str, float]):
        """Record learning for both methods including goal predictions"""
        
        parts = actual_result.split()
        result_char = parts[0].upper()
        score = parts[1] if len(parts) > 1 else None
        
        if result_char not in ['H', 'A', 'D']:
            print(f"⚠️ Invalid result format: {actual_result}")
            return
        
        # Map results
        result_map = {'H': 'home_win', 'A': 'away_win', 'D': 'draw'}
        actual_outcome = result_map[result_char]
        
        # Check enhanced method accuracy
        enhanced_winner = max(enhanced_probs.items(), key=lambda x: x[1])
        enhanced_correct = enhanced_winner[0] == actual_outcome
        
        # Check simplified method accuracy
        simplified_map = {'HOME_WIN': 'home_win', 'AWAY_WIN': 'away_win', 'DRAW': 'draw'}
        simplified_outcome = simplified_map.get(simplified_pred.recommended_bet, 'draw')
        simplified_correct = simplified_outcome == actual_outcome
        
        print(f"\n📚 LEARNING RESULTS")
        print(f"Actual Result: {actual_result}")
        print(f"Enhanced Method: {'✅ CORRECT' if enhanced_correct else '❌ WRONG'}")
        print(f"Simplified Method: {'✅ CORRECT' if simplified_correct else '❌ WRONG'}")
        
        # Goal predictions learning
        if score and '-' in score:
            try:
                home_goals, away_goals = map(int, score.split('-'))
                total_goals = home_goals + away_goals
                both_scored = home_goals > 0 and away_goals > 0
                
                # Check goal predictions
                over_15_pred = goal_probs['over_15'] > goal_probs['under_15']
                over_25_pred = goal_probs['over_25'] > goal_probs['under_25']
                over_35_pred = goal_probs['over_35'] > goal_probs['under_35']
                bts_pred = goal_probs['bts_yes'] > goal_probs['bts_no']
                
                over_15_correct = over_15_pred == (total_goals > 1.5)
                over_25_correct = over_25_pred == (total_goals > 2.5)
                over_35_correct = over_35_pred == (total_goals > 3.5)
                bts_correct = bts_pred == both_scored
                
                print(f"\n⚽ GOAL PREDICTIONS:")
                print(f"Total Goals: {total_goals} | Expected: {goal_probs['total_expected']:.1f}")
                print(f"Over 1.5: {'✅' if over_15_correct else '❌'} | Over 2.5: {'✅' if over_25_correct else '❌'} | Over 3.5: {'✅' if over_35_correct else '❌'}")
                print(f"Both Teams Score: {'✅' if bts_correct else '❌'} (Actual: {'Yes' if both_scored else 'No'})")
                
            except ValueError:
                print(f"⚠️ Could not parse score: {score}")
        
        # Record for enhanced analyzer learning engine
        if hasattr(self.enhanced_analyzer, 'learning_engine'):
            match_id = f"{hash(url)}_{datetime.now().strftime('%Y%m%d')}"
            
            outcome = PredictionOutcome(
                match_id=match_id,
                timestamp=datetime.now(),
                home_team=home_team,
                away_team=away_team,
                predicted_result=result_char if enhanced_correct else ('H' if enhanced_winner[0] == 'home_win' else 'A' if enhanced_winner[0] == 'away_win' else 'D'),
                actual_result=result_char,
                predicted_probabilities=enhanced_probs,
                confidence_score=8 if enhanced_winner[1] > 50 else 5,
                key_factors_used=['form', 'h2h', 'home_advantage'],
                match_characteristics=['merged_analysis'],
                was_correct=enhanced_correct,
                error_magnitude=0 if enhanced_correct else enhanced_winner[1]/100
            )
            
            self.enhanced_analyzer.learning_engine.record_prediction(outcome)
        
        # Simple learning tracking
        try:
            learning_file = "merged_learning.json"
            if Path(learning_file).exists():
                with open(learning_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    'enhanced': {'total': 0, 'correct': 0}, 
                    'simplified': {'total': 0, 'correct': 0},
                    'goals': {'over15': {'total': 0, 'correct': 0}, 'over25': {'total': 0, 'correct': 0}, 
                             'over35': {'total': 0, 'correct': 0}, 'bts': {'total': 0, 'correct': 0}}
                }
            
            data['enhanced']['total'] += 1
            data['simplified']['total'] += 1
            
            if enhanced_correct:
                data['enhanced']['correct'] += 1
            if simplified_correct:
                data['simplified']['correct'] += 1
            
            # Record goal predictions if score available
            if score and '-' in score:
                try:
                    home_goals, away_goals = map(int, score.split('-'))
                    total_goals = home_goals + away_goals
                    both_scored = home_goals > 0 and away_goals > 0
                    
                    # Update goal learning stats
                    for market in ['over15', 'over25', 'over35', 'bts']:
                        data['goals'][market]['total'] += 1
                    
                    if over_15_correct: data['goals']['over15']['correct'] += 1
                    if over_25_correct: data['goals']['over25']['correct'] += 1
                    if over_35_correct: data['goals']['over35']['correct'] += 1
                    if bts_correct: data['goals']['bts']['correct'] += 1
                    
                except ValueError:
                    pass
            
            with open(learning_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            enhanced_acc = data['enhanced']['correct'] / data['enhanced']['total']
            simplified_acc = data['simplified']['correct'] / data['simplified']['total']
            
            print(f"\n📊 Overall Accuracy:")
            print(f"   Enhanced: {enhanced_acc:.1%} ({data['enhanced']['correct']}/{data['enhanced']['total']})")
            print(f"   Simplified: {simplified_acc:.1%} ({data['simplified']['correct']}/{data['simplified']['total']})")
            
            # Show goal prediction accuracy if available
            if data['goals']['over25']['total'] > 0:
                over25_acc = data['goals']['over25']['correct'] / data['goals']['over25']['total']
                bts_acc = data['goals']['bts']['correct'] / data['goals']['bts']['total']
                print(f"   Over 2.5: {over25_acc:.1%} ({data['goals']['over25']['correct']}/{data['goals']['over25']['total']})")
                print(f"   BTS: {bts_acc:.1%} ({data['goals']['bts']['correct']}/{data['goals']['bts']['total']})")
            
        except Exception as e:
            print(f"⚠️ Could not save learning data: {e}")
    
    def analyze_team_news(self, team_name: str, competition: str) -> NewsImpact:
        """Analyze recent team news using AI-powered sentiment analysis"""
        
        news_impact = NewsImpact(team_name=team_name)
        
        try:
            # Search for recent team news
            search_terms = [
                f"{team_name} injury news",
                f"{team_name} lineup team news",
                f"{team_name} latest news {competition}"
            ]
            
            all_news = []
            for term in search_terms:
                news_results = self.fetch_team_news(term)
                all_news.extend(news_results[:3])  # Limit to avoid noise
            
            if not all_news:
                return news_impact
            
            # Analyze news content
            news_impact = self.analyze_news_sentiment(all_news, team_name)
            
        except Exception as e:
            print(f"   ⚠️ News analysis failed for {team_name}: {str(e)[:50]}")
        
        return news_impact
    
    def fetch_team_news(self, search_term: str) -> List[Dict]:
        """Fetch recent team news (simplified implementation)"""
        
        # In a real implementation, this would use news APIs like:
        # - NewsAPI, Google News API, or sports-specific APIs
        # - Web scraping from sports news sites
        # For now, return mock data structure
        
        mock_news = [
            {
                'title': f"Latest updates on {search_term.split()[0]}",
                'content': "Team news and injury updates...",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'sports_news'
            }
        ]
        
        return mock_news
    
    def analyze_news_sentiment(self, news_items: List[Dict], team_name: str) -> NewsImpact:
        """Analyze news sentiment and extract impact factors"""
        
        impact = NewsImpact(team_name=team_name)
        
        # Keywords for different impact categories
        injury_keywords = {
            'negative': ['injured', 'out', 'sidelined', 'surgery', 'ruled out', 'doubt', 'fitness concern'],
            'positive': ['returns', 'fit', 'recovered', 'back', 'available', 'cleared']
        }
        
        lineup_keywords = {
            'negative': ['suspended', 'banned', 'dropped', 'benched', 'rotation'],
            'positive': ['starts', 'key player', 'full strength', 'best eleven', 'strongest lineup']
        }
        
        morale_keywords = {
            'negative': ['crisis', 'pressure', 'criticism', 'poor form', 'struggling', 'disappointing'],
            'positive': ['confident', 'motivated', 'good spirits', 'positive', 'momentum', 'winning streak']
        }
        
        tactical_keywords = {
            'negative': ['tactical problems', 'formation issues', 'system change'],
            'positive': ['tactical improvement', 'new formation', 'tactical advantage']
        }
        
        # Analyze each news item
        all_text = ' '.join([item.get('title', '') + ' ' + item.get('content', '') for item in news_items]).lower()
        
        # Calculate impact scores
        injury_score = self.calculate_keyword_impact(all_text, injury_keywords)
        lineup_score = self.calculate_keyword_impact(all_text, lineup_keywords)
        morale_score = self.calculate_keyword_impact(all_text, morale_keywords)
        tactical_score = self.calculate_keyword_impact(all_text, tactical_keywords)
        
        # Apply impact with bounds
        impact.injury_impact = max(-0.3, min(0.3, injury_score * 0.3))
        impact.lineup_strength = max(-0.2, min(0.2, lineup_score * 0.2))
        impact.morale_factor = max(-0.2, min(0.2, morale_score * 0.2))
        impact.tactical_change = max(-0.1, min(0.1, tactical_score * 0.1))
        
        # Generate news summary
        impact.news_summary = self.generate_news_summary(all_text, team_name)
        
        # Calculate confidence based on news recency and relevance
        impact.confidence = min(1.0, len(news_items) * 0.2 + 0.3)
        
        return impact
    
    def calculate_keyword_impact(self, text: str, keywords: Dict[str, List[str]]) -> float:
        """Calculate impact score based on keyword presence"""
        
        positive_count = sum(1 for keyword in keywords['positive'] if keyword in text)
        negative_count = sum(1 for keyword in keywords['negative'] if keyword in text)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        # Return normalized score between -1 and 1
        return (positive_count - negative_count) / max(1, positive_count + negative_count)
    
    def generate_news_summary(self, text: str, team_name: str) -> List[str]:
        """Generate summary of key news points"""
        
        summary = []
        
        # Check for key injury news
        if any(word in text for word in ['injured', 'out', 'sidelined']):
            summary.append("🏥 Injury concerns reported")
        elif any(word in text for word in ['returns', 'fit', 'recovered']):
            summary.append("✅ Key players returning from injury")
        
        # Check for lineup news
        if any(word in text for word in ['suspended', 'banned', 'dropped']):
            summary.append("⚠️ Key player unavailable")
        elif any(word in text for word in ['full strength', 'best eleven']):
            summary.append("💪 Full strength squad available")
        
        # Check for morale indicators
        if any(word in text for word in ['crisis', 'pressure', 'struggling']):
            summary.append("📉 Team under pressure")
        elif any(word in text for word in ['confident', 'momentum', 'winning']):
            summary.append("📈 Positive team momentum")
        
        return summary[:3]  # Limit to top 3 points

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python merged_analyzer.py <forebet_url> [actual_result]")
        print("Example: python merged_analyzer.py https://www.forebet.com/en/football/matches/team1-team2-12345")
        print("With result: python merged_analyzer.py <url> 'H 2-1'")
        sys.exit(1)
    
    url = sys.argv[1]
    actual_result = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = MergedFootballAnalyzer()
    analyzer.analyze_match(url, actual_result)

if __name__ == "__main__":
    main()
