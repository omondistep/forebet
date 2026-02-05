# Merged Football Analyzer

A comprehensive football match prediction tool that combines two different analysis approaches to provide robust predictions with consensus analysis.

## Overview

The merged analyzer combines:
- **Enhanced Analysis** (from hw.py): Advanced learning system with comprehensive metrics
- **Simplified Predictor** (from predictor.py): Streamlined, reliable prediction model
- **Consensus Analysis**: Compares both methods and highlights agreements/disagreements

## Features

### 🔬 Enhanced Analysis (hw.py method)
- Quality-adjusted form ratings based on opponent strength
- Comprehensive team metrics (25+ factors)
- Head-to-head analysis with learning
- League-specific adjustments
- Manual learning system with prediction tracking
- Advanced probability calculation with 8+ weighted factors

### 🎯 Simplified Predictor (predictor.py method)
- Best-practice probability calculation
- Streamlined team metrics (6 core factors)
- Clear recommendations with reasoning
- Venue-specific form analysis
- Attack vs Defense matchup analysis

### 🤝 Consensus Analysis
- Side-by-side probability comparison
- Agreement/disagreement detection
- Consensus recommendations when methods align
- Average probabilities for reference
- Confidence scoring based on method agreement

## Installation

```bash
# Install dependencies
pip install requests beautifulsoup4 playwright

# Install browser for Playwright
playwright install chromium
```

## Usage

### Basic Analysis
```bash
python merged_analyzer.py <forebet_url>
```

### With Learning (Manual Result Input)
```bash
python merged_analyzer.py <forebet_url> "H 2-1"  # Home win 2-1
python merged_analyzer.py <forebet_url> "A 1-0"  # Away win 1-0
python merged_analyzer.py <forebet_url> "D 1-1"  # Draw 1-1
```

## Output Structure

### 1. Enhanced Analysis Section
```
📊 ENHANCED TEAM METRICS
- Form Rating (quality-adjusted)
- Recent Form (last 3 matches)
- Goals/Match
- Home/Away Performance
- Consistency Score
- vs Top6 Performance

📈 ENHANCED PROBABILITIES
- Detailed probability breakdown
- Enhanced recommendation
```

### 2. Simplified Predictor Section
```
📊 SIMPLIFIED TEAM METRICS
- Recent Form
- Home/Away Form
- Attack/Defense Strength
- Consistency

📈 SIMPLIFIED PROBABILITIES
- Streamlined probability calculation
- Clear recommendation with reasoning
```

### 3. Comparison & Consensus
```
📊 PROBABILITY COMPARISON
- Side-by-side comparison
- Difference analysis

🤝 CONSENSUS ANALYSIS
- Agreement/disagreement detection
- Consensus recommendation
- Confidence scoring
- Average probabilities
```

## Example Output

```
🔍 MERGED FOOTBALL ANALYZER
────────────────────────────────────────────────────────────────────────────────
🎯 Running BOTH Enhanced Analysis (hw.py) AND Simplified Predictor (predictor.py)
────────────────────────────────────────────────────────────────────────────────

⚽ Match: Manchester City vs Liverpool
🏆 Competition: Premier League
📊 Data: 8 home matches, 8 away matches
🤝 H2H: 5 matches found

════════════════════════════════════════════════════════════════════════════════
🔬 ENHANCED ANALYSIS (hw.py method)
════════════════════════════════════════════════════════════════════════════════

📊 ENHANCED TEAM METRICS
Metric                    Home            Away           
-------------------------------------------------------
Form Rating               85.2%           78.1%
Recent Form               90.0%           70.0%
Goals/Match               2.3             1.8
Home/Away Performance     75.0%           65.0%
Consistency               80.0%           70.0%
vs Top6 Performance       60.0%           55.0%

📈 ENHANCED PROBABILITIES
Manchester City  45.2%
Draw            28.1%
Liverpool       26.7%

🎯 Enhanced Recommendation: HOME WIN (45.2%)

════════════════════════════════════════════════════════════════════════════════
🎯 SIMPLIFIED PREDICTOR (predictor.py method)
════════════════════════════════════════════════════════════════════════════════

📊 SIMPLIFIED TEAM METRICS
Metric               Home        Away       
-------------------------------------------
Recent Form          82.5%       75.0%
Home/Away Form       80.0%       60.0%
Attack Strength      2.1         1.9
Defense Strength     1.8         1.6
Consistency          75.0%       70.0%

📈 SIMPLIFIED PROBABILITIES
Manchester City  42.8%
Draw            30.5%
Liverpool       26.7%

🎯 Simplified Recommendation: HOME_WIN (MEDIUM)
💡 Reasoning:
   • Home team favored (42.8%)
   • Home team very strong at home

════════════════════════════════════════════════════════════════════════════════
📊 COMPARISON & CONSENSUS
════════════════════════════════════════════════════════════════════════════════

📊 PROBABILITY COMPARISON
Outcome         Enhanced    Simplified  Difference  
----------------------------------------------------
Manchester Ci   45.2%       42.8%       2.4%
Draw            28.1%       30.5%       2.4%
Liverpool       26.7%       26.7%       0.0%

🤝 CONSENSUS ANALYSIS
✅ BOTH METHODS AGREE: Manchester City Win
📊 Average Probability: 44.0%
🎯 CONSENSUS RECOMMENDATION: Manchester City Win
📈 Consensus Confidence: HIGH

📊 AVERAGE PROBABILITIES
Manchester City  44.0%
Draw            29.3%
Liverpool       26.7%
```

## Learning System

The merged analyzer tracks the accuracy of both methods:

- **Enhanced Method**: Uses advanced learning engine with pattern recognition
- **Simplified Method**: Uses basic accuracy tracking
- **Comparison**: Shows which method performs better over time

Learning data is saved to:
- `prediction_history.json` (Enhanced method)
- `learning_metrics.json` (Enhanced method)
- `merged_learning.json` (Both methods comparison)

## Key Advantages

1. **Dual Perspective**: Two different approaches provide validation
2. **Consensus Detection**: High confidence when methods agree
3. **Disagreement Alerts**: Caution when methods disagree
4. **Learning Integration**: Both methods improve over time
5. **Comprehensive Output**: All relevant information in one analysis

## Files

- `merged_analyzer.py` - Main merged analyzer
- `hw.py` - Original enhanced analyzer
- `predictor.py` - Original simplified predictor
- `test_merged.py` - Test/demo script

## Tips for Best Results

1. **High Confidence**: When both methods agree with similar probabilities
2. **Medium Confidence**: When methods agree but probabilities differ significantly
3. **Low Confidence**: When methods disagree on the outcome
4. **Learning**: Always input actual results to improve both methods
5. **Consensus**: Trust consensus recommendations more than individual method recommendations
