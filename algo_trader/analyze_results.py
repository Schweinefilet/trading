"""
OPTIMIZATION RESULTS ANALYZER
==============================
Interactive analysis of optimization results.

Usage:
    python analyze_results.py optimization_results.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(filepath: str) -> pd.DataFrame:
    """Load optimization results CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} optimization results from {filepath}\n")
    return df


def print_summary(df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    print("\nOBJECTIVES ACHIEVEMENT:")
    print(f"  Win Rate ≥70%: {(df['win_rate'] >= 70.0).sum():,} configs ({(df['win_rate'] >= 70.0).sum() / len(df) * 100:.1f}%)")
    print(f"  Return ≥2147%: {(df['total_return_pct'] >= 2147.71).sum():,} configs ({(df['total_return_pct'] >= 2147.71).sum() / len(df) * 100:.1f}%)")
    print(f"  BOTH: {df['meets_objectives'].sum():,} configs ({df['meets_objectives'].sum() / len(df) * 100:.1f}%)")
    
    print("\nCONSTRAINTS SATISFACTION:")
    print(f"  Trades ≥1200: {(df['total_trades'] >= 1200).sum():,}")
    print(f"  Avg Win ≥$1400: {(df['avg_win'] >= 1400).sum():,}")
    print(f"  Sharpe ≥1.3: {(df['sharpe_ratio'] >= 1.3).sum():,}")
    print(f"  Max DD ≤20%: {(df['max_drawdown_pct'] <= 20.0).sum():,}")
    print(f"  Profit Factor ≥1.5: {(df['profit_factor'] >= 1.5).sum():,}")
    print(f"  ALL: {df['meets_constraints'].sum():,} configs")
    
    print("\nPERFECT CONFIGS (Objectives + Constraints):")
    perfect = df[df['meets_objectives'] & df['meets_constraints']]
    print(f"  Count: {len(perfect)}")
    
    print("\nKEY METRICS DISTRIBUTION:")
    print(f"  Win Rate: {df['win_rate'].min():.1f}% to {df['win_rate'].max():.1f}% (median: {df['win_rate'].median():.1f}%)")
    print(f"  Total Return: {df['total_return_pct'].min():.1f}% to {df['total_return_pct'].max():.1f}% (median: {df['total_return_pct'].median():.1f}%)")
    print(f"  Total Trades: {df['total_trades'].min()} to {df['total_trades'].max()} (median: {df['total_trades'].median():.0f})")
    print(f"  Sharpe Ratio: {df['sharpe_ratio'].min():.2f} to {df['sharpe_ratio'].max():.2f} (median: {df['sharpe_ratio'].median():.2f})")
    print()


def show_top_configs(df: pd.DataFrame, n: int = 10):
    """Show top configurations by different metrics."""
    
    cols = ['config_id', 'win_rate', 'total_return_pct', 'total_trades', 'sharpe_ratio', 'max_drawdown_pct']
    
    print("="*80)
    print(f"TOP {n} CONFIGURATIONS BY WIN RATE")
    print("="*80)
    print(df.nlargest(n, 'win_rate')[cols].to_string(index=False))
    print()
    
    print("="*80)
    print(f"TOP {n} CONFIGURATIONS BY TOTAL RETURN")
    print("="*80)
    print(df.nlargest(n, 'total_return_pct')[cols].to_string(index=False))
    print()
    
    print("="*80)
    print(f"TOP {n} CONFIGURATIONS BY SHARPE RATIO")
    print("="*80)
    print(df.nlargest(n, 'sharpe_ratio')[cols].to_string(index=False))
    print()
    
    # Composite score: balance win rate and return
    df['composite_score'] = (df['win_rate'] / 70.0) * (df['total_return_pct'] / 2147.71)
    print("="*80)
    print(f"TOP {n} BY COMPOSITE SCORE (Win Rate × Total Return)")
    print("="*80)
    print(df.nlargest(n, 'composite_score')[cols + ['composite_score']].to_string(index=False))
    print()


def analyze_parameter_impact(df: pd.DataFrame):
    """Analyze which parameters have the most impact."""
    print("="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)
    
    # Identify parameter columns
    param_cols = [c for c in df.columns if c not in [
        'config_id', 'win_rate', 'total_return_pct', 'total_trades', 'avg_win', 'avg_loss',
        'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 'profit_factor', 'largest_win',
        'largest_loss', 'annualized_return', 'expectancy', 'meets_objectives', 'meets_constraints',
        'objective_score', 'constraint_penalty', 'compromise_score', 'composite_score'
    ]]
    
    correlations = []
    for param in param_cols:
        if df[param].dtype in [np.float64, np.int64]:
            corr_wr = df[param].corr(df['win_rate'])
            corr_ret = df[param].corr(df['total_return_pct'])
            correlations.append({
                'parameter': param,
                'win_rate_corr': corr_wr,
                'return_corr': corr_ret,
                'abs_impact': abs(corr_wr) + abs(corr_ret)
            })
    
    impact_df = pd.DataFrame(correlations).sort_values('abs_impact', ascending=False)
    
    print("\nTop 15 Most Impactful Parameters:")
    print(impact_df.head(15).to_string(index=False))
    print()


def find_best_compromises(df: pd.DataFrame, n: int = 10):
    """Find best compromise configurations (near objectives with good constraints)."""
    print("="*80)
    print("BEST COMPROMISE CONFIGURATIONS")
    print("="*80)
    
    # Score: how close to objectives while meeting constraints
    df['objective_score'] = (df['win_rate'] / 70.0) + (df['total_return_pct'] / 2147.71)
    df['constraint_penalty'] = 0
    df.loc[df['total_trades'] < 1200, 'constraint_penalty'] += 1
    df.loc[df['avg_win'] < 1400, 'constraint_penalty'] += 1
    df.loc[df['sharpe_ratio'] < 1.3, 'constraint_penalty'] += 1
    df.loc[df['max_drawdown_pct'] > 20.0, 'constraint_penalty'] += 1
    df.loc[df['profit_factor'] < 1.5, 'constraint_penalty'] += 1
    
    df['compromise_score'] = df['objective_score'] - (df['constraint_penalty'] * 0.2)
    
    cols = ['config_id', 'win_rate', 'total_return_pct', 'total_trades', 'sharpe_ratio', 
            'objective_score', 'constraint_penalty', 'compromise_score']
    
    best = df.nlargest(n, 'compromise_score')
    print(best[cols].to_string(index=False))
    print()


def export_top_configs(df: pd.DataFrame, output_prefix: str = 'top_configs'):
    """Export detailed parameters for top configurations."""
    
    # Top 10 by win rate
    top_wr = df.nlargest(10, 'win_rate')
    top_wr.to_csv(f'{output_prefix}_win_rate.csv', index=False)
    
    # Top 10 by return
    top_ret = df.nlargest(10, 'total_return_pct')
    top_ret.to_csv(f'{output_prefix}_return.csv', index=False)
    
    # Perfect configs
    perfect = df[df['meets_objectives'] & df['meets_constraints']]
    if len(perfect) > 0:
        perfect.to_csv(f'{output_prefix}_perfect.csv', index=False)
    
    print(f"✓ Exported top configurations to {output_prefix}_*.csv\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze optimization results')
    parser.add_argument('results_file', type=str, help='Path to optimization_results.csv')
    parser.add_argument('--top', type=int, default=10, help='Number of top configs to show')
    parser.add_argument('--export', type=str, default='top_configs', 
                        help='Prefix for exported top config files')
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results_file)
    
    # Run analyses
    print_summary(df)
    show_top_configs(df, args.top)
    analyze_parameter_impact(df)
    find_best_compromises(df, args.top)
    export_top_configs(df, args.export)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
