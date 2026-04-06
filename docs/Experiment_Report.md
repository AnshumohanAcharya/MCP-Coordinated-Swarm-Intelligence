# Experiment Report — Final Review
## MCP-Coordinated Swarm Intelligence

> Run `make final-review` to regenerate all numbers below.  
> Results are saved to `results/final_review/summary_report.json` and the
> six-panel PNG figure.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Environment | 100×100 grid (1 km × 1 km), wildfire + flood zones |
| UAVs | 5 (Experiments 1 & 2); 3/5/7/10 (Experiment 3) |
| Steps per episode | 10 000 |
| Battery drain rate | 0.0001/s (~10 % over 10 000 steps) |
| UAV max speed | 15.0 m/s |
| Sensor range | 50.0 m |
| Episodes (full mode) | 20 (failure) / 15 (comm) / 12 (scalability) |
| Random seeds | BASE_SEED + ep (paired, reproducible) |
| Primary metric | Priority-Weighted Coverage at mission end (%) |
| Statistical test | Paired t-test, 95 % confidence interval |

---

## Comparison Design

Each episode uses the **same fixed seed** for both MCP and Baseline arms,
ensuring the environment is identical. This paired design maximises the power
of the t-test because within-pair variance is eliminated.

**MCP arm** (`MCPExplorer`):
- UAVs divided into n horizontal sectors
- Shared coverage map via MCP broadcast
- Failed-UAV detection: MCP context reveals absent neighbours

**Baseline arm** (`BaselineExplorer`):
- Same greedy coverage algorithm
- No cross-UAV information sharing
- Unaware of neighbour coverage or failures

---

## Experiment 1 — UAV Failure Resilience

### Scenarios

| Scenario | Failures | Failure steps |
|---|---|---|
| No failure | 0 | — |
| 1 failure | 1 (UAV 0) | Step 2 000 |
| 2 failures | 2 (UAV 0, 1) | Steps 2 000, 6 000 |

### Expected Results

Under no-failure conditions, both arms should achieve similar coverage
(validates that the baseline algorithm itself is sound). The difference emerges
when failures occur:

- **MCP under 2 failures**: surviving UAVs detect the failed sectors via
  shared context and expand coverage boundaries to compensate. Expected: ≥ 2 %
  priority-weighted coverage advantage over baseline.
- **Baseline under 2 failures**: no redistribution mechanism. Sectors of failed
  UAVs remain surveyed only up to the failure point. Expected: coverage drops
  proportionally to the fraction of unsurveyed sector remaining.

### Statistical Significance

For each failure level, a paired t-test across episodes produces:
- t-statistic
- p-value (< 0.05 = statistically significant, < 0.01 = highly significant)
- 95 % CI for MCP − Baseline difference

---

## Experiment 2 — Communication Degradation

### Protocol

Message delivery probability tested at: **1.0, 0.8, 0.6, 0.4, 0.2**

At reliability r, each MCP message is independently dropped with probability
1 − r. When messages are dropped, the agent uses its last received shared map
(stale context).

### Expected Results

| Reliability | Expected MCP advantage |
|---|---|
| 1.0 | Maximum advantage (perfect comms) |
| 0.8 | Slight reduction; stale maps infrequent |
| 0.6 | Moderate reduction; threshold region |
| 0.4 | Advantage may not be significant |
| 0.2 | MCP and Baseline likely converge |

The curve shows how **gracefully** MCP degrades — even with 40 % packet loss,
shared context from the previous 10 steps is still useful for sector assignment.

---

## Experiment 3 — Scalability Study

### Swarm Sizes

Tested: **3, 5, 7, 10 UAVs**

MCP message complexity: each UAV sends 1 broadcast per step → n × steps total.
Context aggregation is a single-pass summation → O(n) per step.

### Expected Results

| Metric | Expected |
|---|---|
| context_agg_time_ms | Linear growth with n |
| Coverage per UAV | Decreases (diminishing returns) as n grows |
| MCP vs Baseline Δ | Largest at n=5–7 (when overlap is most wasteful without coordination) |

The O(n) result is the strongest algorithmic claim: it proves MCP is scalable
to large swarms, unlike centralised planners which require O(n²) pairwise
negotiation.

---

## GPS-Denied SLAM Validation

### Model

Position noise follows a decaying Gaussian:

```
σ(t) = σ_final + (σ_init - σ_final) · max(0, 1 - t / 5000)
     = 1.0 + 7.0 · max(0, 1 - t / 5000)  metres
```

At step 0: σ = 8 m (UAV just initialised, SLAM map sparse)  
At step 5000: σ = 1 m (SLAM converged after ~500 m of flight)

The noise is applied to the position the agent *uses to pick its next target*,
not to the physical position. Physical movement and sensor coverage are still
computed from the true UAV state.

### Expected Results

- Coverage drop under GPS-denied vs normal: < 3 % (because noise decays quickly)
- MCP under GPS-denied ≥ Baseline under GPS-denied (sector anchoring still helps)
- At high noise (early in mission), MCP advantage slightly reduced but restored
  as SLAM converges

Run with: `python run_final_review_demo.py --mode full --gps-denied`

---

## Generating Results

```bash
# Full publication-quality run
make final-review

# GPS-denied mode
make gps-denied-test

# Quick sanity check
make final-review-quick
```

Output files:
- `results/final_review/summary_report.json` — all metrics as JSON
- `results/final_review/combined_figure.png` — six-panel figure for presentation
- `results/final_review/failure_resilience/` — per-scenario JSON and plots
- `results/final_review/communication_degradation/` — per-reliability JSON
- `results/final_review/scalability/` — per-swarm-size JSON

---

## Interpreting the Six-Panel Figure

| Panel | Content |
|---|---|
| Top-left | Coverage over time: MCP vs Baseline (no failure) |
| Top-centre | Coverage over time under 2-failure scenario |
| Top-right | Active UAV fraction over time (shows failure events) |
| Bottom-left | Coverage vs communication reliability |
| Bottom-centre | Context aggregation time vs swarm size |
| Bottom-right | Coverage per UAV vs swarm size |
