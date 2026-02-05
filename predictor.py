#!/usr/bin/env python3
"""
Improved Football Prediction Model
Incorporates best practices in prediction logic with manual learning
"""

import sys
import re
import json
import pickle
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from collections import namedtuple
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics

# Core data structures
Match = namedtuple('Match', ['date', 'home_team', 'away_team', 'score', 'result', 'goals_scored', 'goals_conceded', 'venue', 'opponent_position'])

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

class DataExtractor:
    """Reuses working extraction logic from hw.py"""
    
    def __init__(self):
        self.team_cache = {}
        self.league_standings = {}
    
    def fetch_with_playwright(self, url: str) -> Optional[str]:
        """Fetch page using Playwright with requests fallback"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-blink-features=AutomationControlled']
                )
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.new_page()
                
                # Block unnecessary resources
                page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,eot,css}", 
                          lambda route: route.abort())
                
                response = page.goto(url, timeout=15000, wait_until='domcontentloaded')
                if response and response.ok:
                    page.wait_for_timeout(1000)
                    html = page.content()
                    browser.close()
                    return html
                
                browser.close()
                return None
        except Exception as e:
            print(f"   ⚠️ Playwright failed, trying requests fallback...")
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=15)
                if response.ok:
                    return response.text
            except Exception as e2:
                print(f"   ❌ Requests also failed: {str(e2)[:50]}")
            return None
    
    def clean_team_name(self, name: str) -> str:
        """Clean team name - borrowed from hw.py"""
        if not name or not isinstance(name, str):
            return ""
        
        name = re.sub(r'onmouseover\s*=\s*["\']showhint\([^)]*\)["\']', '', name, flags=re.IGNORECASE)
        name = re.sub(r'onclick\s*=\s*["\'][^"\']*["\']', '', name, flags=re.IGNORECASE)
        name = re.sub(r'<[^>]+>', '', name)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def extract_team_names_and_positions(self, soup: BeautifulSoup) -> Tuple[str, str, Optional[int], Optional[int]]:
        """Extract team names and positions - borrowed from hw.py"""
        home_team, away_team = None, None
        home_position, away_position = None, None
        
        # Try header selectors
        header_selectors = ['div.rcnt > strong', 'h1.team-name', 'div.team_info strong']
        
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
        
        # Fallback to title
        if not home_team or not away_team:
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                match = re.search(r'([^\-]+)\s+vs\s+([^\-]+)\s+Prediction', title_text, re.IGNORECASE)
                if match:
                    home_team = self.clean_team_name(match.group(1))
                    away_team = self.clean_team_name(match.group(2))
        
        # Extract standings for positions
        standings = self.extract_standings(soup)
        if home_team:
            home_position = self.get_position_from_standings(standings, home_team)
        if away_team:
            away_position = self.get_position_from_standings(standings, away_team)
        
        return home_team, away_team, home_position, away_position
    
    def extract_standings(self, soup: BeautifulSoup) -> Dict[str, int]:
        """Extract league standings - simplified from hw.py"""
        standings = {}
        
        # Look for standings tables
        tables = soup.find_all('table')
        for table in tables:
            text = table.get_text()
            if any(keyword in text.lower() for keyword in ['pts', 'position', 'rank']):
                rows = table.find_all("tr")
                for row in rows[1:]:
                    cols = row.find_all("td")
                    if len(cols) >= 3:
                        pos_text = cols[0].get_text(strip=True)
                        if pos_text.isdigit():
                            position = int(pos_text)
                            for col in cols[1:]:
                                col_text = col.get_text(strip=True)
                                if col_text and len(col_text) > 2 and not col_text.isdigit():
                                    team_name = self.clean_team_name(col_text)
                                    if team_name:
                                        standings[team_name] = position
                                        break
        
        return standings
    
    def get_position_from_standings(self, standings: Dict[str, int], team_name: str) -> Optional[int]:
        """Get team position from standings"""
        if not standings or not team_name:
            return None
        
        # Direct match
        for team, pos in standings.items():
            if team.lower() == team_name.lower():
                return pos
        
        # Partial match
        for team, pos in standings.items():
            if team_name.lower() in team.lower() or team.lower() in team_name.lower():
                return pos
        
        return None
    
    def extract_recent_matches(self, html: str, team_name: str, count: int = 6) -> List[Match]:
        """Extract recent matches - simplified from hw.py"""
        matches = []
        clean_team = self.clean_team_name(team_name)
        
        # Pattern for match extraction
        patterns = [
            r'(\d{2}/\d{2}/\d{4})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            r'([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
        ]
        
        for pattern in patterns:
            found_matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
            
            for match_tuple in found_matches:
                try:
                    if len(match_tuple) == 5:
                        date, team1, g1, g2, team2 = match_tuple
                    elif len(match_tuple) == 4:
                        team1, g1, g2, team2 = match_tuple
                        date = ""
                    else:
                        continue
                    
                    team1_clean = self.clean_team_name(team1.strip())
                    team2_clean = self.clean_team_name(team2.strip())
                    
                    g1_int = int(g1.strip())
                    g2_int = int(g2.strip())
                    
                    # Determine if our team is involved
                    if clean_team.lower() in team1_clean.lower():
                        result = "W" if g1_int > g2_int else "D" if g1_int == g2_int else "L"
                        venue = 'H'
                        goals_scored = g1_int
                        goals_conceded = g2_int
                        opponent_position = self.get_position_from_standings(self.league_standings, team2_clean)
                    elif clean_team.lower() in team2_clean.lower():
                        result = "W" if g2_int > g1_int else "D" if g2_int == g1_int else "L"
                        venue = 'A'
                        goals_scored = g2_int
                        goals_conceded = g1_int
                        opponent_position = self.get_position_from_standings(self.league_standings, team1_clean)
                    else:
                        continue
                    
                    match_obj = Match(
                        date=date,
                        home_team=team1_clean,
                        away_team=team2_clean,
                        score=f"{g1_int}-{g2_int}",
                        result=result,
                        goals_scored=goals_scored,
                        goals_conceded=goals_conceded,
                        venue=venue,
                        opponent_position=opponent_position
                    )
                    matches.append(match_obj)
                    
                except (ValueError, IndexError):
                    continue
        
        return matches[:count]
    
    def extract_h2h_data(self, soup: BeautifulSoup, home_team: str, away_team: str) -> MatchContext:
        """Extract head-to-head data"""
        context = MatchContext()
        
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
                    
                    if score1 > score2:
                        context.h2h_home_wins += 1
                    elif score2 > score1:
                        context.h2h_away_wins += 1
                    else:
                        context.h2h_draws += 1
                    
                    matches_found += 1
                    if matches_found >= 10:  # Limit to avoid noise
                        break
                except:
                    continue
        
        context.h2h_total = context.h2h_home_wins + context.h2h_away_wins + context.h2h_draws
        return context

class ImprovedPredictor:
    """Improved prediction model with best practices"""
    
    def __init__(self):
        self.extractor = DataExtractor()
        self.learning_data = self.load_learning_data()
    
    def load_learning_data(self) -> Dict:
        """Load learning data from hw.py if available"""
        try:
            learning_file = Path("../learning_metrics.json")
            if learning_file.exists():
                with open(learning_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {'total_predictions': 0, 'accuracy': 0.0}
    
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
    
    def compute_team_metrics(self, team_name: str, matches: List[Match], position: Optional[int]) -> TeamForm:
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
    
    def calculate_probabilities(self, home_form: TeamForm, away_form: TeamForm, context: MatchContext) -> Prediction:
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
            reasoning=reasoning
        )
    
    def analyze_match(self, url: str, silent: bool = False) -> Optional[Prediction]:
        """Main analysis function"""
        if not silent:
            print(f"\n🔍 IMPROVED FOOTBALL PREDICTOR")
            print("─" * 50)
            print(f"📡 Fetching: {url[:60]}...")
        
        # Fetch and parse page
        html = self.extractor.fetch_with_playwright(url)
        if not html:
            if not silent:
                print("❌ Failed to fetch page")
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract basic info
        home_team, away_team, home_pos, away_pos = self.extractor.extract_team_names_and_positions(soup)
        if not home_team or not away_team:
            if not silent:
                print("❌ Could not extract team names")
            return None
        
        if not silent:
            print(f"⚽ Match: {home_team} vs {away_team}")
        
        # Extract match data
        home_matches = self.extractor.extract_recent_matches(html, home_team, 6)
        away_matches = self.extractor.extract_recent_matches(html, away_team, 6)
        h2h_context = self.extractor.extract_h2h_data(soup, home_team, away_team)
        
        if not silent:
            print(f"📊 Data: {len(home_matches)} home matches, {len(away_matches)} away matches")
            if h2h_context.h2h_total > 0:
                print(f"🤝 H2H: {h2h_context.h2h_total} matches found")
        
        # Compute team metrics
        home_form = self.compute_team_metrics(home_team, home_matches, home_pos)
        away_form = self.compute_team_metrics(away_team, away_matches, away_pos)
        
        # Generate prediction
        prediction = self.calculate_probabilities(home_form, away_form, h2h_context)
        
        # Store team names in prediction for web app
        prediction.home_team = home_team
        prediction.away_team = away_team
        
        # Display results only if not silent
        if not silent:
            self.display_prediction(home_team, away_team, home_form, away_form, prediction)
        
        return prediction
    
    def display_prediction(self, home_team: str, away_team: str, 
                          home_form: TeamForm, away_form: TeamForm, 
                          prediction: Prediction):
        """Display prediction results"""
        
        print(f"\n{'═' * 50}")
        print(f"🎯 PREDICTION RESULTS")
        print(f"{'═' * 50}")
        
        # Team metrics
        print(f"\n📊 TEAM METRICS")
        print(f"{'Metric':<20} {'Home':<12} {'Away':<12}")
        print("-" * 45)
        print(f"Recent Form          {home_form.recent_form:.1%}        {away_form.recent_form:.1%}")
        print(f"Home/Away Form       {home_form.home_form:.1%}        {away_form.away_form:.1%}")
        print(f"Attack Strength      {home_form.attack_strength:.1f}          {away_form.attack_strength:.1f}")
        print(f"Defense Strength     {home_form.defense_strength:.1f}          {away_form.defense_strength:.1f}")
        print(f"Consistency          {home_form.consistency:.1%}        {away_form.consistency:.1%}")
        
        # Probabilities
        print(f"\n📈 MATCH PROBABILITIES")
        print(f"{home_team[:15]:<15} {prediction.home_prob:5.1f}%")
        print(f"{'Draw':<15} {prediction.draw_prob:5.1f}%")
        print(f"{away_team[:15]:<15} {prediction.away_prob:5.1f}%")
        
        # Recommendation
        print(f"\n🎯 RECOMMENDATION")
        if prediction.recommended_bet:
            bet_map = {
                'HOME_WIN': f"{home_team[:12]} Win",
                'AWAY_WIN': f"{away_team[:12]} Win", 
                'DRAW': 'Draw'
            }
            print(f"✅ BET: {bet_map.get(prediction.recommended_bet, prediction.recommended_bet)}")
            print(f"📊 Confidence: {prediction.confidence}")
            
            if prediction.reasoning:
                print(f"💡 Reasoning:")
                for reason in prediction.reasoning:
                    print(f"   • {reason}")
        else:
            # This should never happen now
            print(f"❌ NO BET RECOMMENDED")
            print(f"📊 Confidence: {prediction.confidence}")
            print(f"💡 Reason: System error - no recommendation generated")
        
        print(f"\n{'─' * 50}")
        print(f"📚 Manual Learning: Add result with 'H 2-1', 'A 1-0', or 'D 1-1'")
        print(f"{'─' * 50}")
    
    def record_result(self, prediction: Prediction, actual_result: str):
        """Record actual result for learning"""
        result_map = {'H': 'HOME_WIN', 'A': 'AWAY_WIN', 'D': 'DRAW'}
        actual = result_map.get(actual_result.upper())
        
        if actual and prediction.recommended_bet:
            was_correct = (prediction.recommended_bet == actual)
            
            # Update learning data
            self.learning_data['total_predictions'] = self.learning_data.get('total_predictions', 0) + 1
            if was_correct:
                self.learning_data['correct_predictions'] = self.learning_data.get('correct_predictions', 0) + 1
            
            accuracy = self.learning_data.get('correct_predictions', 0) / self.learning_data['total_predictions']
            self.learning_data['accuracy'] = accuracy
            
            print(f"\n📚 LEARNING UPDATE")
            print(f"Predicted: {prediction.recommended_bet}, Actual: {actual}")
            print(f"Result: {'✅ CORRECT' if was_correct else '❌ WRONG'}")
            print(f"Overall Accuracy: {accuracy:.1%} ({self.learning_data.get('correct_predictions', 0)}/{self.learning_data['total_predictions']})")
            
            # Save learning data
            try:
                with open('learning_data.json', 'w') as f:
                    json.dump(self.learning_data, f, indent=2)
            except:
                pass

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <forebet_url> [actual_result]")
        print("Example: python predictor.py https://www.forebet.com/en/football/matches/team1-team2-12345")
        print("With result: python predictor.py <url> 'H 2-1'")
        sys.exit(1)
    
    url = sys.argv[1]
    actual_result = sys.argv[2] if len(sys.argv) > 2 else None
    
    predictor = ImprovedPredictor()
    prediction = predictor.analyze_match(url)
    
    if prediction and actual_result:
        # Parse result
        parts = actual_result.split()
        if parts and parts[0].upper() in ['H', 'A', 'D']:
            predictor.record_result(prediction, parts[0])

if __name__ == "__main__":
    main()
