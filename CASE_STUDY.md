# Cutting LLM API costs by 82% without flying blind on quality

**I built a routing layer that sends each prompt to the cheapest model capable of handling
it, then verifies the answer against a frontier model and automatically escalates when it
falls short.** On an 800-request load test it cut spend **82.5%** against an all-GPT-4o
baseline, routed **86.9%** of held-out prompts to the correct tier, and caught the
under-routed remainder with an async verification loop.

---

## The problem

Most LLM applications send every request to their best model. It's the safe default, and
it's the reason inference bills look the way they do.

But traffic isn't uniform. In a coding assistant, "What does `len()` do?" and "Design a
sharded rate limiter tolerant of clock drift" arrive through the same endpoint. One of
them needs a frontier model. The other needs almost nothing.

The naive fix — let users pick a model — doesn't work. Users pick wrong, and they pick
based on price rather than need. The routing decision has to be automatic, and it has to
be **safe when it's wrong**, because it will sometimes be wrong.

---

## What I built

### 1. Complexity classification

A RandomForest over 12 engineered prompt features (keyword density across three
difficulty vocabularies, constraint counts, output-format signals, instruction depth)
assigns each prompt to one of three tiers. Tiers map to models in a YAML config, so
swapping a model is a config change, not a deploy:

```yaml
tier_1: { primary: llama3.2,    fallback: claude-haiku-4-5 }   # local, free
tier_2: { primary: gpt-4o-mini, fallback: claude-haiku-4-5 }   # cheap
tier_3: { primary: gpt-4o,      fallback: claude-sonnet-4-6 }  # frontier
```

### 2. Async verification against a frontier reference

This is the part that makes cheap routing defensible. After the response is returned, a
background worker sends **the same prompt** to the top-tier model and compares the two
outputs. How they're compared depends on what the request actually was:

| Use case | Bar | Comparison |
|---|---|---|
| Extraction | 4.5 / 5 | Did it capture every field the reference found? |
| Classification | 5.0 / 5 | Does its label match the reference label exactly? |
| Summarization | 4.0 / 5 | LLM-as-judge rubric |
| Code generation | 3.8 / 5 | Same rubric, lower bar |
| Explanation | 3.5 / 5 | Same rubric |
| General | 3.5 / 5 | Semantic agreement with the reference |

A single quality threshold would have been wrong. "Good enough" for a summary is a judge
score; "good enough" for a classification is an exact label match. Grading them the same
way means either failing good summaries or passing wrong labels.

### 3. Escalation with a latency budget

A failed verification re-runs the request one tier up. If the better answer arrives inside
the latency budget it's served; otherwise it's logged for the audit trail and the caller
collects it from `GET /v1/completions/{id}/escalation`.

**Escalation spend is counted against the savings figure, not excluded from it.** This
matters: the naive calculation on this run reads 84.1%, because it only counts the first
call. Counting the re-runs gives 82.5%. I'd rather publish the smaller honest number.

### 4. The feedback loop

Every routing failure is a labelled example the classifier got wrong — this prompt was
routed to tier N and the output didn't hold up, so it belonged higher. Failures accumulate
with an inferred corrected tier (a severe quality gap jumps two tiers, a marginal one
jumps one), deduplicated by prompt hash, and a background worker retrains weekly.

The prompts the classifier misjudges are exactly the ones it learns from.

```
 prompt ──▶ classify ──▶ cheapest capable model ──▶ response to user
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              compare against frontier model
                          │
                    failed? ─── yes ──▶ escalate  ──▶ better answer
                          │                    └────▶ training example
                          no                              │
                          ▼                               ▼
                      done                      weekly retrain
```

---

## Results

800 held-out prompts, 30 concurrent, coding-assistant traffic mix (50% simple / 35%
moderate / 15% complex):

| Metric | Result |
|---|---|
| **Cost reduction** | **82.5%** |
| Spend vs baseline | $0.42 vs $2.37 |
| Routing accuracy (held-out) | 86.9% |
| Under-routed (sent too cheap) | 6.1% |
| Throughput | 30.4 req/s |
| Latency p95 | 1,448 ms |
| Traffic served without a frontier model | 85.6% |

Routing accuracy by tier: **86.6%** simple, **84.2%** moderate, **94.4%** complex. The
classifier is most reliable exactly where mistakes are most expensive.

![Cost dashboard](reports/screenshots/dashboard-hero.png)

_The operational view: spend against the counterfactual, routing distribution, quality
distribution, and escalation rate over time._

### Savings depend on your traffic, not on my router

This is the number most cost-routing write-ups get wrong. Savings are a property of the
**traffic mix**, so one headline figure is close to meaningless on its own:

| Traffic mix | Savings |
|---|---:|
| Light (70/25/5) | 91.7% |
| **Typical (50/35/15)** | **81.9%** |
| Uniform (33/33/33) | 67.8% |
| Heavy (25/35/40) | 63.4% |

Same system, same code, 63%–92% depending entirely on what you send it. Any cost-routing
claim quoted without its traffic mix should be treated as unverified.

---

## What I can prove, and what I can't

A cost-reduction number is easy to inflate and hard to trust. Here's the honest accounting.

**Defensible:**
- **82.5% cost reduction** — arithmetic on published per-token prices, escalation spend
  included, traffic mix stated.
- **86.9% routing accuracy** — measured on prompts held out from training, whose correct
  tier is known by construction.
- **Length-leak elimination** — an earlier version of this classifier reported *100%
  accuracy* because the training corpus encoded difficulty in prompt length (tier 1 capped
  at 63 characters, tier 3 started at 191; a single length threshold scored 93%). I
  rebuilt the corpus so length carries no signal. A permutation test now puts length-only
  accuracy at 37.7% against a 33.6% shuffled baseline, **p = 0.118 — indistinguishable
  from chance.** Three regression tests guard it.

**Not yet provable:**
- **Quality parity.** The load test ran against mock providers, so verifier pass rates
  measure my simulation's constants, not real model quality. I won't quote a parity figure
  I can't support. Establishing one requires real API keys and a human-labelled set.
- **Token counts.** Simulated in mock mode, so cost figures are guesses about
  tokenization, not billing.
- **Real-traffic accuracy.** 86.9% is against my held-out pool. Live prompts are messier.

**What would close the gap:** an A/B test — half of live traffic through the router, half
straight to the frontier model, compared on cost *and* a blind quality rating of the same
prompts. That's the only design that supports a causal claim, and it's the next thing I'd
build.

---

## Engineering decisions worth noting

**Verification is off the request path.** Users get the cheap answer at cheap-model
latency; verification and escalation happen behind them. Quality assurance that adds a
frontier-model round trip to every request defeats the purpose.

**The router doesn't import the verifier.** The verifier depends on the logger, which
depends on the router's response type. Wiring them directly would cycle, so the engine
exposes a completion hook and a composition root connects them. This also means the
routing path is testable without any verification machinery.

**Config validated before it's applied.** `PUT /v1/routing-config` checks every model
against the live registry first, so a typo can't leave the router pointing at a model that
doesn't exist.

**The metric that inflates is the one to scrutinize.** I found two bugs in my own savings
calculation: escalation spend wasn't counted at all, and once it was, I initially summed
the wrong quantity (a *delta* that can go negative, which would have credited the system
for failing). Both made the headline look better. Both are now covered by tests.

---

## Further reading

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — request lifecycle, component responsibilities, why the router cannot import the verifier, and the cost-accounting rules
- **[SECURITY.md](SECURITY.md)** — the defects found in review and what remains
- **[reports/load_test_report.md](reports/load_test_report.md)** — the full 800-request run

## Stack

Python 3.11 · FastAPI · scikit-learn · SQLAlchemy/SQLite · Streamlit + Plotly · Docker
Compose · pytest (111 tests)

Providers: OpenAI, Anthropic, Ollama — behind one adapter interface, with a mock mode for
development without keys.

```bash
docker compose up --build     # API :8000, dashboard :8501, worker
python scripts/load_test.py --requests 800
python scripts/significance_analysis.py   # which numbers are real
```
