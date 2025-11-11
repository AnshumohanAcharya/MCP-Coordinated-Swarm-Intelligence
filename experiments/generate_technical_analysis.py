#!/usr/bin/env python3
"""
Generate comprehensive technical analysis reports from experimental results.
This script analyzes baseline comparison data and generates detailed technical reports.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import statistics

def load_results(json_path: str) -> Dict[str, Any]:
    """Load experimental results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate comprehensive statistics for a dataset."""
    if not data:
        return {}
    
    data_array = np.array(data)
    return {
        'mean': float(np.mean(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'median': float(np.median(data_array)),
        'q25': float(np.percentile(data_array, 25)),
        'q75': float(np.percentile(data_array, 75)),
        'cv': float(np.std(data_array) / np.mean(data_array) * 100) if np.mean(data_array) != 0 else 0.0
    }

def analyze_performance_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance metrics from experimental results."""
    baseline_results = results.get('baseline_results', [])
    mcp_results = results.get('mcp_results', [])
    
    # Extract metrics
    baseline_rewards = [r['reward'] for r in baseline_results]
    mcp_rewards = [r['reward'] for r in mcp_results]
    
    baseline_coverage = [r['coverage'] for r in baseline_results]
    mcp_coverage = [r['coverage'] for r in mcp_results]
    
    baseline_battery = [r['battery_efficiency'] for r in baseline_results]
    mcp_battery = [r['battery_efficiency'] for r in mcp_results]
    
    baseline_collisions = [r['collision_count'] for r in baseline_results]
    mcp_collisions = [r['collision_count'] for r in mcp_results]
    
    # Calculate statistics
    analysis = {
        'rewards': {
            'baseline': calculate_statistics(baseline_rewards),
            'mcp': calculate_statistics(mcp_rewards),
            'improvement_percent': ((np.mean(mcp_rewards) - np.mean(baseline_rewards)) / np.mean(baseline_rewards) * 100) if np.mean(baseline_rewards) != 0 else 0.0
        },
        'coverage': {
            'baseline': calculate_statistics(baseline_coverage),
            'mcp': calculate_statistics(mcp_coverage),
            'improvement_percent': ((np.mean(mcp_coverage) - np.mean(baseline_coverage)) / np.mean(baseline_coverage) * 100) if np.mean(baseline_coverage) != 0 else 0.0
        },
        'battery': {
            'baseline': calculate_statistics(baseline_battery),
            'mcp': calculate_statistics(mcp_battery),
            'improvement_percent': ((np.mean(mcp_battery) - np.mean(baseline_battery)) / np.mean(baseline_battery) * 100) if np.mean(baseline_battery) != 0 else 0.0
        },
        'collisions': {
            'baseline': calculate_statistics(baseline_collisions),
            'mcp': calculate_statistics(mcp_collisions),
            'improvement_percent': ((np.mean(baseline_collisions) - np.mean(mcp_collisions)) / np.mean(baseline_collisions) * 100) if np.mean(baseline_collisions) != 0 else 0.0
        }
    }
    
    return analysis

def generate_latex_table(analysis: Dict[str, Any], metric_name: str) -> str:
    """Generate LaTeX table for a specific metric."""
    metric_data = analysis[metric_name]
    baseline = metric_data['baseline']
    mcp = metric_data['mcp']
    improvement = metric_data['improvement_percent']
    
    table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{{metric_name.capitalize()} Performance Comparison}}
\\label{{tab:{metric_name}}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Baseline}} & \\textbf{{MCP-Coordinated}} & \\textbf{{Improvement}} \\\\
\\midrule
Mean & {baseline['mean']:.3f} & {mcp['mean']:.3f} & {improvement:+.1f}\\% \\\\
Standard Deviation & {baseline['std']:.3f} & {mcp['std']:.3f} & {((mcp['std'] - baseline['std']) / baseline['std'] * 100):+.1f}\\% \\\\
Minimum & {baseline['min']:.3f} & {mcp['min']:.3f} & - \\\\
Maximum & {baseline['max']:.3f} & {mcp['max']:.3f} & - \\\\
Median & {baseline['median']:.3f} & {mcp['median']:.3f} & - \\\\
Coefficient of Variation & {baseline['cv']:.2f}\\% & {mcp['cv']:.2f}\\% & {((mcp['cv'] - baseline['cv']) / baseline['cv'] * 100):+.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return table

def generate_technical_report(results_path: str, output_path: str):
    """Generate comprehensive technical analysis report."""
    results = load_results(results_path)
    analysis = analyze_performance_metrics(results)
    
    report = f"""
# Technical Analysis Report
Generated from: {results_path}

## Executive Summary

This report provides comprehensive technical analysis of experimental results comparing
baseline (Independent PPO) and MCP-coordinated swarm approaches.

## Performance Metrics Analysis

### Reward Performance
- Baseline Average: {analysis['rewards']['baseline']['mean']:.2f} (SD: {analysis['rewards']['baseline']['std']:.2f})
- MCP-Coordinated Average: {analysis['rewards']['mcp']['mean']:.2f} (SD: {analysis['rewards']['mcp']['std']:.2f})
- Improvement: {analysis['rewards']['improvement_percent']:+.2f}%

### Coverage Performance
- Baseline Average: {analysis['coverage']['baseline']['mean']:.3f}% (SD: {analysis['coverage']['baseline']['std']:.3f})
- MCP-Coordinated Average: {analysis['coverage']['mcp']['mean']:.3f}% (SD: {analysis['coverage']['mcp']['std']:.3f})
- Improvement: {analysis['coverage']['improvement_percent']:+.2f}%
- Coefficient of Variation: Baseline {analysis['coverage']['baseline']['cv']:.2f}% vs MCP {analysis['coverage']['mcp']['cv']:.2f}%

### Battery Efficiency
- Baseline Average: {analysis['battery']['baseline']['mean']:.2f}% (SD: {analysis['battery']['baseline']['std']:.2f})
- MCP-Coordinated Average: {analysis['battery']['mcp']['mean']:.2f}% (SD: {analysis['battery']['mcp']['std']:.2f})
- Improvement: {analysis['battery']['improvement_percent']:+.2f}%

### Collision Analysis
- Baseline Average: {analysis['collisions']['baseline']['mean']:.2f} collisions/episode
- MCP-Coordinated Average: {analysis['collisions']['mcp']['mean']:.2f} collisions/episode
- Improvement: {analysis['collisions']['improvement_percent']:+.2f}%

## Key Findings

1. **Consistency**: MCP-coordinated agents show {((analysis['coverage']['baseline']['cv'] - analysis['coverage']['mcp']['cv']) / analysis['coverage']['baseline']['cv'] * 100):.1f}% lower coefficient of variation in coverage, indicating more stable performance.

2. **Battery Efficiency**: MCP-coordinated agents achieve {analysis['battery']['improvement_percent']:+.2f}% better battery efficiency with {((analysis['battery']['baseline']['std'] - analysis['battery']['mcp']['std']) / analysis['battery']['baseline']['std'] * 100):.1f}% lower variance.

3. **Peak Performance**: MCP-coordinated agents achieve higher peak rewards ({analysis['rewards']['mcp']['max']:.2f} vs {analysis['rewards']['baseline']['max']:.2f}).

## LaTeX Tables

{generate_latex_table(analysis, 'rewards')}

{generate_latex_table(analysis, 'coverage')}

{generate_latex_table(analysis, 'battery')}

{generate_latex_table(analysis, 'collisions')}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Technical analysis report generated: {output_path}")

if __name__ == "__main__":
    results_path = "results/baseline_comparison.json"
    output_path = "results/technical_analysis_report.md"
    
    if Path(results_path).exists():
        generate_technical_report(results_path, output_path)
    else:
        print(f"Error: Results file not found at {results_path}")

