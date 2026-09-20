# Load Test Report

_Generated 2026-09-16T20:08:45.986038+00:00_

## 82.5% cost reduction

**$0.4156 spent** against a **$2.3733** all-top-tier baseline across **800 requests**.

| | |
|---|---|
| Routed-model spend | $0.3770 |
| Escalation spend (re-runs) | $0.0386 |
| **Total** | **$0.4156** |
| Baseline (all top-tier) | $2.3733 |
| **Saved** | **$1.9576** |

## Throughput and latency

| Metric | Value |
|---|---|
| Requests | 800 |
| Concurrency | 30 |
| Throughput | 30.1 req/s |
| Routing wall time | 26.6s |
| Latency p50 | 1024 ms |
| Latency p95 | 1454 ms |
| Latency p99 | 1493 ms |

## Routing distribution

| Model | Requests | Share |
|---|---:|---:|
| `llama3.2` | 407 | 50.9% |
| `gpt-4o-mini` | 278 | 34.8% |
| `gpt-4o` | 115 | 14.4% |

## Routing accuracy (measured, not simulated)

Every prompt in the pool was written for a known tier and is held out from the classifier's training data, so this is a real accuracy figure — unlike the verifier pass rate, which is simulated in mock mode.

**86.9% of requests reached the tier they were written for.**

| Intended tier | n | Routed correctly |
|---|---:|---:|
| simple | 419 | 86.6% |
| moderate | 273 | 84.2% |
| complex | 108 | 94.4% |

**6.1% were under-routed** (49 requests sent to a model below the tier they needed). This is the direction that matters: under-routing inflates the savings figure while degrading the answer. The async verifier exists to catch exactly these.

## Sensitivity to traffic mix

This run assumed a **typical** mix (50% simple / 35% moderate / 15% complex). Savings depend far more on the mix than on sampling error, so the same system produces very different headlines on different workloads:

| Traffic mix | Savings |
|---|---:|
| typical | 81.9%  ← this run |
| light | 91.7% |
| heavy | 63.4% |
| uniform | 67.8% |

## What these numbers are

- **Cost figures are arithmetic** on published per-token prices from `config/models.yaml`, applied to simulated token counts.
- **Savings describe the traffic mix**, not the router alone — see the table above.
- **Quality/pass/escalation rates are simulated** when running with mock providers, and measure the mock rather than any real model.
- Prompts are **held out** from the classifier's training corpus.

Run `python scripts/significance_analysis.py` for the full breakdown of which metrics in this project are measurements and which are not.
