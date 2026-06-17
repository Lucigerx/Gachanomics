# The Gacha-Nomics

**EC424 Capstone · Thammasat University, Faculty of Economics**  
*Behavioral Economics × Agent-Based Simulation × Modern Portfolio Theory*

---

## The Question

Young Thai workers spend on mobile games, livestream shopping, betting, K-pop — nothing that looks dangerous on its own. But some stay in control and some don't. What actually separates them?

## The Finding

It is not the people who spend on many different things who spiral into debt. It is the people who concentrate deeply on **one habit**.

Agents with a Herfindahl-Hirschman Index (HHI) above 0.5 spiral at roughly **58× the rate** of diversified agents (HHI < 0.3), stable across 10 random seeds. Adding an explicit Prospect Theory loss-aversion layer sharpens the gradient to **128×**.

> **Depth over breadth** — behavioral concentration, not breadth, is the primary driver of household debt spirals among employed urban Thai Gen Z.

---

## Project Structure

```
gachanomics/
├── GachaNomics.ipynb       ← Full Colab notebook (primary submission)
├── gachanomics_v6.py       ← Standalone script (numbers only, no plots)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The **notebook** is the canonical version — all visualizations, robustness checks, and inline citations live there. The **standalone script** reproduces the core numerical results without matplotlib or Colab dependencies.

---

## Quick Start

**Run in Google Colab (recommended)**

Open `GachaNomics.ipynb` and run all cells top to bottom. No API key required — `USE_LLM_CALIBRATION = False` by default.

**Run the standalone script**

```bash
pip install -r requirements.txt
python gachanomics_v6.py
```

Expected runtime: ~5 seconds (pure NumPy, no LLM calls).

**Optional — LLM-assisted damage calibration**

```bash
export ANTHROPIC_API_KEY=your_key_here
# Set USE_LLM_CALIBRATION = True in gachanomics_v6.py
python gachanomics_v6.py
```

Cost: ~$0.016 · ~3 minutes · Claude Haiku · ~6 API calls.

---

## How the Model Works

The simulation runs in five phases:

| Phase | What it does |
|---|---|
| 1 — Market returns | 60-month synthetic return series calibrated to Max-Sharpe QQQ/SPY/GLD/TLT (2019–2024), with 2 VIX-shock months injected |
| 2 — Event calendar | 60×7 real-world spending trigger matrix — gacha banner cycles, TikTok mega-sales (9.9/10.10/11.11/12.12), EPL/World Cup betting seasons, K-pop concerts, lottery dates |
| 3 — LLM validation | Optional one-time Claude Haiku validation of BASE_DMG anchors (±30% of real-world sourced values) |
| 4A — Version A | 10,000 hybrid agents; each gets a 7-dimensional behavioral weight vector drawn from Dirichlet(α), anchored to Thai Gen Z marginal prevalence |
| 4B — Version B | 6 pure archetypes (Mu / K-Pop / TikTok / Saver / Betting / Gacha), 3,000 paths each — the interpretable control group |
| 5 — Wealth engine | Single-pool simulation: market return → DCA deposit → debt service → impulse spend → BNPL compound interest |

**Activation function:** `p_fire = 1 − exp(−β · score)` where `score = (DNA · EVENT) / DNA.sum()`

The guardrail (`/ DNA.sum()`) normalizes total event intensity equally across all agent types, so concentration effects are not confounded by how many axes an agent activates. Beta is calibrated per-axis to reproduce published Thai Gen Z baseline impulse probabilities.

**Debt spiral flag:** `debt > 3 × monthly_DCA` — emergent output, not fitted to any target.

---

## Key Parameters

| Parameter | Value | Source |
|---|---|---|
| BNPL monthly rate | 1.33% | BOT ceiling 16%/yr (Aug 2020) |
| Loss-chase multiplier | 1.75× | CGS Chulalongkorn 2023: 85.2% of indebted bettors continue gambling |
| Dirichlet α | [0.73, 0.27, 0.77, 0.55, 0.11, 0.40, 0.60] | Thai Gen Z marginal prevalence (author-set; α sensitivity sweep in notebook) |
| λ (loss aversion) | [1.90, 1.955, 1.80, 1.40, 1.955, 1.40, 2.20] per axis | Brown et al. 2024 base = 1.955; Betting low by design — risk-seeking in the loss domain |
| K-Pop whale event | ~฿18,000 | Bangkok Post 2022 (Jackson Wang VIP floor) |
| Betting whale event | ~฿12,000 | MGR Online / Kantar 2024 (World Cup mega-bet) |
| Gacha whale event | ~฿8,000 | Codashop TH 2025 (pity ceiling) |
| Portfolio benchmark | μ = 1.10%/mo · σ = 4.17%/mo | Max-Sharpe QQQ/SPY/GLD/TLT 2019–2024 |

---

## Results

**Baseline: τ = 0.15, seed = 42**

Version B — pure archetypes

| Archetype | Spiral % | Final wealth (฿) | Gap vs MPT |
|---|---|---|---|
| Mu | 0.0% | 160,056 | 36.4% |
| K-Pop | 0.0% | 234,037 | 41.9% |
| TikTok | 0.4% | 187,471 | 62.8% |
| Saver | 0.0% | 746,626 | 5.6% |
| Betting | 0.0% | 133,670 | 55.7% |
| Gacha | 0.0% | 225,616 | 36.0% |

Version A — 10,000 hybrid agents (10-seed CI)

| Metric | Value |
|---|---|
| Mean terminal wealth | ฿287,283 |
| Overall spiral rate | 1.96% ± 0.12 |
| Mean wealth gap vs MPT | 35.6% ± 0.3 |
| Diversified spiral (HHI < 0.3) | 0.15% ± 0.06 |
| Concentrated spiral (HHI > 0.5) | 8.53% ± 0.84 |
| Concentration gradient | ~58× |
| With explicit λ (Prospect Theory) | ~128× (gradient sharpens; overall spiral rate drops) |

---

## Robustness Checks

Five checks, all reported in the notebook:

1. **VIX-mode** — suppress (×0.7) vs amplify (×2.0); depth-over-breadth direction holds under both
2. **Whale-magnitude sweep** — 0.5×–2.0×; depth gradient holds in the realistic range
3. **Multi-seed CI** — 10 seeds; spiral range 1.73–2.19%, gradient range 7.46–9.83%
4. **Explicit λ (Prospect Theory)** — overall spiral drops but concentration gradient sharpens to 128×
5. **Guardrail ablation** — removing the score normalization guardrail does not invert the depth-over-breadth finding

---

## Scope and Honest Limitations

**Population:** Employed urban Thai Gen Z, ages 22–28, income ฿25,000–35,000/mo. Not representative of students or low-income groups — age-stratified recalibration is Future Work.

**TikTok sensitivity:** The TikTok BASE_DMG of ฿14,000 is a mega-sale event basket (Kantar 2024), not average monthly spend. At regular monthly spend (~฿2,000–5,000), spiral rates drop to 0%. All results are explicitly scoped to heavy and event-driven shoppers.

**DNA independence:** Behavioral axes are drawn independently from the Dirichlet prior. Real inter-axis co-occurrence (e.g. do TikTok heavy shoppers also bet?) is not modeled — primary survey data needed.

**Single-pool wealth engine:** A non-fungible 3-bucket cascade (Liquid → Buffer → Locked) per Thaler 1985 would raise spiral rates. Documented as Future Work; the current single-pool is a deliberate scope decision, not an oversight.

**Gacha scope:** Standard resetting pity only. Non-resetting variants (e.g. HSR Epitomized Path) are Future Work.

**Spiral rates are not incidence rates.** They are relative comparisons within the model's scope — not predictions of real-world debt events in the Thai population.

---

## Future Work

1. **Primary survey (n ≥ 300)** — replace author-set Dirichlet α with measured Thai Gen Z behavioral co-occurrence data
2. **Age-stratified recalibration** — replace author-set AXIS_INIT and AXIS_DCA with Krungsri Gen Z Finance Survey 2025 and BOT deposit distribution data
3. **3-bucket mental accounting engine** — Thaler 1985 cascade (Liquid → Buffer → Locked) as a non-fungible wealth engine
4. **Out-of-sample validation** — calibrate on survey split A, validate concentration→spiral prediction on split B
5. **Guardrail ablation panel** — 4-way visualization (B / A-guarded / A-unguarded / A-no-loss-chase)

---

## Data Sources

| Source | Used for |
|---|---|
| Bank of Thailand — BNPL circular (Aug 2020) | BNPL monthly rate (1.33%) |
| CGS Chulalongkorn University (2023) | Betting loss-chase multiplier (85.2% re-bet rate) |
| Kantar TGI Thailand / Bangkok Post (2024) | TikTok mega-sale basket (฿14,000); archetype segmentation |
| Bangkok Post (2022) | K-Pop whale event floor — Jackson Wang VIP (฿18,000) |
| The MATTER / MGR Online (2023) | K-Pop average concert ticket (฿5,270) |
| Statista Mobile Games Market Forecast (2024) | Thai mobile gaming ARPU ($105.80/yr per user) |
| Antom / Ant Group Thailand Gaming Report (2025) | Per-payer spend ($393/yr); 49% paying-user base |
| Codashop Thailand (2025) | Competitive gaming BASE_DMG and whale ceiling |
| Brown et al. (2024) | Loss-aversion lambda baseline (1.955) |
| Markowitz (1952) / Yahoo Finance (2019–2024) | MPT Max-Sharpe benchmark construction |

---

## Techniques

| Technique | Role in this project |
|---|---|
| Agent-Based Modeling (ABM) | Heterogeneous agents; emergent debt-spiral behavior |
| Dirichlet distribution | Behavioral DNA vectors; concentration vs. diversification |
| HHI (Herfindahl-Hirschman Index) | Repurposed from market-concentration → behavioral concentration (core of RQ2) |
| Modern Portfolio Theory (MPT) | Max-Sharpe rational benchmark; wealth gap = behavioral cost (RQ1) |
| Prospect Theory / loss aversion | Explicit λ robustness check — not in the main engine to avoid double-counting |
| Mental accounting (Thaler 1985) | Motivates single-pool scope; 3-bucket cascade = Future Work |
| Probabilistic activation | `p = 1 − exp(−β·score)` — hazard-style, calibrated to published impulse rates |
| Monte Carlo (multi-seed CI) | Seed-variance quantification across 10 seeds |
| Sensitivity and ablation analysis | τ sweep, whale-magnitude sweep, guardrail ablation, α sensitivity |
| Vectorized NumPy | 10,000 agents × 60 months as matrix operations — no agent-level loops |

---

## Citation

```
Jiratdechakul, P. (2026). The Gacha-Nomics: An Agent-Based Model of
Impulse-Spending and Household Debt Dynamics Among Thai Gen Z.
EC424 Capstone, Faculty of Economics, Thammasat University.
Advisor: Prof. Thiraphap Fakthong.
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Advisor: Prof. Thiraphap Fakthong · Faculty of Economics, Thammasat University*
