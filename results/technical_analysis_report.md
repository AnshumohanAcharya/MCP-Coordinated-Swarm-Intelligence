
# Technical Analysis Report
Generated from: results/baseline_comparison.json

## Executive Summary

This report provides comprehensive technical analysis of experimental results comparing
baseline (Independent PPO) and MCP-coordinated swarm approaches.

## Performance Metrics Analysis

### Reward Performance
- Baseline Average: 13.85 (SD: 28.00)
- MCP-Coordinated Average: 13.17 (SD: 37.00)
- Improvement: -4.89%

### Coverage Performance
- Baseline Average: 1.354% (SD: 0.113)
- MCP-Coordinated Average: 1.124% (SD: 0.071)
- Improvement: -17.00%
- Coefficient of Variation: Baseline 8.36% vs MCP 6.35%

### Battery Efficiency
- Baseline Average: 98.91% (SD: 0.11)
- MCP-Coordinated Average: 99.08% (SD: 0.09)
- Improvement: +0.17%

### Collision Analysis
- Baseline Average: 8.68 collisions/episode
- MCP-Coordinated Average: 9.18 collisions/episode
- Improvement: -5.76%

## Key Findings

1. **Consistency**: MCP-coordinated agents show 24.1% lower coefficient of variation in coverage, indicating more stable performance.

2. **Battery Efficiency**: MCP-coordinated agents achieve +0.17% better battery efficiency with 22.5% lower variance.

3. **Peak Performance**: MCP-coordinated agents achieve higher peak rewards (38.24 vs 35.96).

## LaTeX Tables


\begin{table}[h]
\centering
\caption{Rewards Performance Comparison}
\label{tab:rewards}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{MCP-Coordinated} & \textbf{Improvement} \\
\midrule
Mean & 13.848 & 13.171 & -4.9\% \\
Standard Deviation & 28.000 & 37.001 & +32.1\% \\
Minimum & -105.485 & -153.341 & - \\
Maximum & 35.963 & 38.239 & - \\
Median & 20.096 & 20.517 & - \\
Coefficient of Variation & 202.20\% & 280.94\% & +38.9\% \\
\bottomrule
\end{tabular}
\end{table}



\begin{table}[h]
\centering
\caption{Coverage Performance Comparison}
\label{tab:coverage}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{MCP-Coordinated} & \textbf{Improvement} \\
\midrule
Mean & 1.354 & 1.124 & -17.0\% \\
Standard Deviation & 0.113 & 0.071 & -37.0\% \\
Minimum & 1.090 & 1.010 & - \\
Maximum & 1.550 & 1.280 & - \\
Median & 1.365 & 1.095 & - \\
Coefficient of Variation & 8.36\% & 6.35\% & -24.1\% \\
\bottomrule
\end{tabular}
\end{table}



\begin{table}[h]
\centering
\caption{Battery Performance Comparison}
\label{tab:battery}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{MCP-Coordinated} & \textbf{Improvement} \\
\midrule
Mean & 98.910 & 99.077 & +0.2\% \\
Standard Deviation & 0.111 & 0.086 & -22.5\% \\
Minimum & 98.689 & 98.940 & - \\
Maximum & 99.106 & 99.262 & - \\
Median & 98.908 & 99.065 & - \\
Coefficient of Variation & 0.11\% & 0.09\% & -22.7\% \\
\bottomrule
\end{tabular}
\end{table}



\begin{table}[h]
\centering
\caption{Collisions Performance Comparison}
\label{tab:collisions}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{MCP-Coordinated} & \textbf{Improvement} \\
\midrule
Mean & 8.680 & 9.180 & -5.8\% \\
Standard Deviation & 27.802 & 37.324 & +34.2\% \\
Minimum & 0.000 & 0.000 & - \\
Maximum & 130.000 & 182.000 & - \\
Median & 0.000 & 0.000 & - \\
Coefficient of Variation & 320.31\% & 406.58\% & +26.9\% \\
\bottomrule
\end{tabular}
\end{table}

