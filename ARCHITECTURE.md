# Architecture

How a request moves through LLM Cost AutoPilot, why the pieces are separated the way
they are, and what each one is responsible for.

The system solves one problem: **send each prompt to the cheapest model that can handle
it, and be safe when that judgement is wrong.** Almost every design decision below
follows from the second half of that sentence.

---

## 1. Layering

Dependencies point in one direction. Nothing below reaches upward.

```
  ┌──────────────────────────────────────────────────────────┐
  │  api/          HTTP surface: auth, validation, schemas    │
  └───────────────────────────┬──────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────┐
  │  service.py    composition root — the only module that    │
  │                knows routing and verification coexist     │
  └──────────┬──────────────────────────────┬────────────────┘
             │                              │
  ┌──────────▼─────────────┐   ┌────────────▼─────────────────┐
  │  router/               │   │  verifier/                   │
  │   engine, escalation   │   │   worker, judge, use_cases,  │
  │                        │   │   feedback                   │
  └──────────┬─────────────┘   └────────────┬─────────────────┘
             │                              │
  ┌──────────▼──────────────────────────────▼─────────────────┐
  │  classifier/     models/          logging/                 │
  │   tiers          registry         database                 │
  │   features       providers        logger                   │
  │   model          tokens, mode                              │
  └────────────────────────────────────────────────────────────┘
```

**The router does not import the verifier.** This is the one structural constraint worth
stating outright, because the obvious design violates it:

```
verifier/worker  →  logging/logger  →  router/engine     (existing, necessary)
router/engine    →  verifier/worker                      (would close the cycle)
```

`logger` needs `RoutedResponse` to write a log row, so it imports the engine. If the
engine then imported the worker to queue verification, the import graph would cycle.
Instead the engine exposes a **completion hook** ([engine.py:32](src/router/engine.py:32))
and `service.py` connects the two ends. The payoff beyond avoiding the cycle: the routing
path is testable with no verification machinery present at all.

---

## 2. Request lifecycle

A single `POST /v1/completions` traverses two phases. The first is synchronous and ends
when the caller gets their answer. The second runs behind them.

### Phase 1 — synchronous (the caller waits)

```
1. authenticate            api/security.py      constant-time key comparison
2. validate                api/schemas.py       size caps, role checks
3. context-window check    api/security.py      reject what no model could fit
4. classify                classifier/model.py  12 features → RandomForest → tier
5. map tier → model        router/engine.py     routing.yaml lookup
6. call provider           models/providers/    with fallback on failure
7. compute counterfactual  router/engine.py     cost on the priciest model
8. fire completion hook    service.py           log, then queue verification
9. return                                       answer + routing metadata
```

Step 7 is what makes the savings figure possible: every response carries what it *would*
have cost had it gone to the most expensive model, so the counterfactual is recorded per
request rather than estimated later.

Step 8 is deliberately last and deliberately guarded — a failure in logging or queueing
must never reach a caller who already has a valid answer
([engine.py:184](src/router/engine.py:184)).

### Phase 2 — asynchronous (off the request path)

```
VerificationWorker._process_job
  │
  ├── 1. detect use case        verifier/use_cases.py   → picks the policy
  ├── 2. fetch reference        verifier/judge.py       → same prompt, top-tier model
  ├── 3. score                  verifier/judge.py       → strategy from the policy
  ├── 4. persist outcome        logging/logger.py       → request_logs
  │
  └── if it failed:
        ├── 5. escalate         router/escalation.py    → one tier up, latency-gated
        ├── 6. record event     logging/database.py     → escalation_events
        └── 7. record feedback  verifier/feedback.py    → labelled training example
```

Only a sampled fraction reaches this phase (`verification.sample_rate`), because step 2
costs a full frontier-model call.

---

## 3. Components

### classifier/

Turns a prompt into a tier. `features.py` extracts 12 numeric features:

| # | Feature | # | Feature |
|---|---|---|---|
| 0 | `token_count` | 6 | `moderate_keyword_score` |
| 1 | `char_count` | 7 | `complex_keyword_score` |
| 2 | `sentence_count` | 8 | `has_code_block` |
| 3 | `question_count` | 9 | `has_context` |
| 4 | `constraint_count` | 10 | `output_complexity` |
| 5 | `simple_keyword_score` | 11 | `instruction_depth` |

A RandomForest maps those to `SIMPLE | MODERATE | COMPLEX`. Engineered features rather
than embeddings: classification sits on the critical path, so it has to be fast and
explainable — every routing decision returns the reasoning that produced it.

> **A caution encoded in the tests.** Features 0–2 are length proxies. An earlier corpus
> had tier correlated with prompt length, and the classifier learned "long = complex"
> while reporting 100% accuracy. `tests/test_training_data.py` now fails if a
> length-only model scores above chance. See [SECURITY.md](SECURITY.md) and
> `scripts/significance_analysis.py`.

### models/

Provider abstraction. `BaseProvider.send_request()` is the entire contract; every adapter
returns the same `LLMResponse`, so the router, logger and verifier are provider-agnostic.

- `registry.py` — models and prices from `config/models.yaml`; also answers "which is
  cheapest" and "which is highest quality", which the counterfactual depends on
- `mode.py` — decides mock vs live, and **raises rather than silently mocking** when live
  mode is misconfigured
- `tokens.py` — `tiktoken` counting for paths where the provider reports no usage

### router/

`engine.py` orchestrates the synchronous phase. `escalation.py` owns the re-run: it
walks one tier up, tries that tier's primary then fallback within `max_retries`, and
gates serving the improved answer on a latency budget.

### verifier/

The quality layer, and the reason cheap routing is defensible.

`use_cases.py` detects what kind of request this is and selects a policy. `judge.py`
fetches a reference response from the top-tier model and scores against it using the
strategy the policy names:

| Use case | Bar | Strategy | What it compares |
|---|---:|---|---|
| Extraction | 4.5 | `field_coverage` | Fields in the reference that the answer also captured |
| Classification | 5.0 | `label_match` | Normalised label equality |
| Summarization | 4.0 | `rubric` | LLM-as-judge on accuracy/completeness/coherence |
| Code generation | 3.8 | `rubric` | Same rubric, lower bar |
| Explanation | 3.5 | `rubric` | Same rubric |
| General | 3.5 | `agreement` | Token overlap plus a length-ratio penalty |

One global threshold would be wrong in both directions: it would fail acceptable
summaries and pass incorrect classifications. All six bars are overridable from
`config/routing.yaml` without touching code.

When a comparison cannot be made — the response already came from the reference model,
or the reference produced nothing parseable — the outcome is **inconclusive**, scored
exactly at the threshold. It neither triggers escalation nor counts as evidence the
routing was right. Failing a response for our own inability to measure it would be worse
than not measuring.

`feedback.py` closes the loop. A failure is an example the classifier got wrong, so it is
stored with a corrected tier inferred from the size of the quality gap (a severe miss
jumps two tiers), deduplicated by prompt hash, and merged into the training set on a
weekly retrain.

### logging/

`database.py` holds the schema and every aggregate query. Two tables:

**`request_logs`** — one row per request: identity (`prompt_hash`, `prompt_preview`),
routing (`complexity_tier`, `routed_model`, `use_case`), economics (`cost`,
`hypothetical_cost`, `input_tokens`, `output_tokens`), and verification
(`quality_score`, `quality_threshold`, `quality_gap`, `divergence`,
`verification_strategy`, `was_verified`).

**`escalation_events`** — one row per escalation, kept separate so `cost_delta` and
`quality_gap` survive in full rather than being flattened onto the request row. Carries
`original_model`, `escalated_model`, both costs, the triggering reason, and whether the
better answer made it inside the latency budget.

### service.py

The composition root, and the only place that knows routing and verification are parts of
one system. It builds every component, registers the completion hook, owns the retrain
scheduler, and holds a bounded LRU of escalated results for callers to collect.

---

## 4. Cost accounting

The headline number is computed in one place and deliberately understates itself.

```
routed_cost      = Σ request_logs.cost                 first attempt
escalation_cost  = Σ escalation_events.escalated_cost  re-runs
total_cost       = routed_cost + escalation_cost
baseline         = Σ request_logs.hypothetical_cost    all-frontier counterfactual
savings %        = (baseline − total_cost) / baseline
```

Two things this gets right that are easy to get wrong:

**Escalation spend counts against the savings.** Re-running a failed request is a real
cost of operating the router. Excluding it inflated the figure by 1.6 points on the load
test.

**The additional cost is the re-run's full cost, not its delta.** An escalated request is
paid for twice. Summing `cost_delta` would credit the system whenever the escalated model
happened to be cheaper — rewarding failure.

Savings vary far more with **traffic mix** than with anything the router does (63%–92%
across mixes on the same code), so every report states the mix it assumed.

---

## 5. Configuration

Three surfaces, none requiring a redeploy:

| File / surface | Controls |
|---|---|
| `config/models.yaml` | Model catalogue: prices, context windows, quality tiers |
| `config/routing.yaml` | Tier→model mapping, per-use-case bars, escalation, feedback cadence |
| `PUT /v1/routing-config` | Runtime tier remapping, validated against the registry first |

The runtime endpoint validates every referenced model before applying anything, so a typo
cannot leave the router pointing at a model that does not exist.

---

## 6. Deployment

```
 docker compose up
   ├── api        uvicorn + in-process verification worker      :8000
   ├── worker     periodic classifier retraining                 —
   └── dashboard  streamlit                                     :8501
        └── all three share one SQLite volume (WAL + busy timeout)
```

**Verification runs inside the API process**, not the worker container. The queue is an
in-memory `asyncio.Queue`, so another container cannot consume it. What *is* separable is
retraining — CPU-bound sklearn work that belongs off the request path — so that is what
the `worker` service does. Scaling verification across processes would require a real
broker; the honest trade was made in favour of zero infrastructure dependencies.

---

## 7. Decisions and trade-offs

**Verification is off the request path.** Users get the cheap answer at cheap-model
latency. Quality assurance that added a frontier round trip to every request would defeat
the purpose. The cost: the better answer arrives *after* the response, so it is collected
by polling rather than pushed.

**Sampling, not exhaustive verification.** Verifying everything doubles spend. The
`sample_rate` knob trades confidence for cost.

**Engineered features over embeddings.** Faster, explainable, and debuggable — at a
ceiling on accuracy (71.8% ± 6.6% cross-validated).

**SQLite over Postgres.** One file, no service to run, WAL for concurrent readers.
Correct for a single host; swap `DATABASE_URL` before real traffic.

**Mock mode is explicit-only.** It must be asked for. Inferring it from a missing key
meant a misconfigured deployment served fabricated answers labelled as real ones.

---

## 8. Extension points

| To add… | Touch |
|---|---|
| A model | `config/models.yaml` — nothing else, if its provider exists |
| A provider | Subclass `BaseProvider`, register in `RoutingEngine._providers` |
| A use case | `UseCase` enum, detection patterns, and a policy in `DEFAULT_POLICIES` |
| A scoring strategy | `VerificationStrategy` enum plus a `_score_*` method on `QualityJudge` |
| A tier | `ComplexityTier`, `routing.yaml`, and retrain with the new labels |

---

## 9. Constraints

- Verification cannot scale beyond one process without a broker
- SQLite is single-host
- No rate limiting; payload caps bound a single request, not a caller
- The classifier is trained on a hand-authored corpus, so accuracy on live traffic is
  unmeasured
- Escalated answers are polled, not pushed

Further reading: [README](README.md) · [CASE_STUDY](CASE_STUDY.md) ·
[SECURITY](SECURITY.md)
