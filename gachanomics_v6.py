"""
================================================================================
THE GACHA-NOMICS V6 -- Context-Activated DNA Model (Deadline Build)
================================================================================
EC424 Capstone | Quantitative Finance x Behavioral Economics x Agent-Based Sim
Author: Phumiphat Jiratdechakul

Hybrid LLM-ABM architecture:
  Phase 1 (numpy)  : market returns -- monthly return simulation calibrated to real market data
  Phase 2 (numpy)  : 60x7 real-world Event Vector calendar (the V6 contribution)
  Phase 3 (Claude) : OPTIONAL one-time LLM validation of BASE_DMG (Scenario 2)
  Phase 4 (numpy)  : two agent layers (A: Dirichlet DNA, B: 6 archetypes)
  Phase 5 (numpy)  : shared single-pool wealth/debt engine + A-vs-B comparison

CALIBRATION (validated):
  * Activation is PROBABILISTIC: p_fire = 1 - exp(-beta * score). An event fires
    stochastically each month; it is NOT a guaranteed full-damage hit.
  * Per-axis BETA is calibrated so pure-type archetypes reproduce published baseline impulse
    probabilities (Mu .45, K-pop .30, TikTok .55, Saver .12, Betting .45, Gacha .30).
  * Debt-spiral is an EMERGENT output of real-world whale-event magnitudes
    (K-pop floor/Labubu ~B22k, Betting WC mega-bet ~B12k). It is NOT fitted to
    any pre-specified target. Under this single-pool engine the emergent spiral is lower than
    parametric benchmarks -- a reported finding, not a target.

SCOPE / HONEST LABELS (for the paper):
  * DNA vectors drawn INDEPENDENTLY from a Dirichlet prior anchored to published
    Thai marginal prevalence. Inter-axis correlation (co-occurrence) is NOT
    modeled -- primary-survey future work.
  * Wealth engine uses a SINGLE liquid pool. A non-fungible 3-bucket
    mental-accounting cascade (which would drive higher spiral rates) is documented as future work;
    BY REFERENCE, not re-implemented. A demonstration confirmed non-restrictive
    buckets are behaviorally identical to a single pool -- isolating bucket
    NON-FUNGIBILITY as the specific source of the H3 effect. H3 therefore is not
    re-tested in this model; single-pool is a scope decision.
  * Gacha = STANDARD RESETTING pity only. Non-resetting variants (HSR Epitomized
    Path, constellation chasing) = future work.
  * Betting fragility is driven by a LOSS-CHASING multiplier, NOT a high lambda.
    lambda_betting is intentionally LOW (risk-seeking in the loss domain).

Cost: Scenario 2 (USE_LLM_CALIBRATION=True) ~ $0.016 / ~3 min (Haiku, ~6 calls).
      USE_LLM_CALIBRATION=False -> $0 / ~5s pure-numpy.
================================================================================
"""

import numpy as np
import os, json, time

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
np.random.seed(SEED)

N_MONTHS  = 60
N_AGENTS  = 10_000      # Version A hybrid population
N_PATHS_B = 3_000       # Version B paths per archetype

# Axis order (FIXED everywhere): 0:Mu 1:K-Pop 2:TikTok 3:Gacha 4:Comp 5:Bet 6:Saver
AXES = ["Mu", "K-Pop", "TikTok", "Gacha", "Competitive", "Betting", "Saver"]
MU, KPOP, TIKTOK, GACHA, COMP, BET, SAVER = range(7)

# Loss-aversion lambda per axis (Brown et al. 2024 base = 1.955).
# Betting LOW on purpose; chasing handled by LOSS_CHASE_MULT, not lambda.
LAMBDA_AXIS = np.array([1.90, 1.955, 1.80, 1.40, 1.955, 1.40, 2.20])

# Base expected damage per EVENT, THB (real-world sourced).
BASE_DMG  = np.array([500., 4500., 14000., 1500., 800., 1633., 0.])
DMG_SIGMA = 0.60

# Whale tail: rare large SINGLE events, REAL THB magnitudes + monthly prob.
#   K-pop floor ticket / Labubu ~B22k (MGR/Kantar); Betting WC mega-bet ~B12k;
#   Gacha pity-dump ~B8k. Spiral emerges from these, not from fitting.
WHALE_THB  = np.array([3000., 22000., 14000., 8000., 4000., 12000., 0.])
WHALE_PROB = np.array([0.010, 0.040,  0.020,  0.030, 0.010, 0.050, 0.000])

# Engine constants
BNPL_RATE       = 0.0133    # 1.33%/mo = BOT BNPL ceiling 16%/yr (Aug 2020)
MIN_PAYMENT_RATE = 0.05     # 5% min payment, floor B300
DEBT_SPIRAL_K   = 3.0       # debt > 3x monthly DCA = spiral
LOSS_CHASE_MULT = 1.75      # Betting re-bet mult after a loss month (CGS 85.2%)

# Published baseline impulse rates -- used ONLY to calibrate per-axis beta, not spiral.
BASELINE_IMPULSE_P = {"Mu":.45, "K-pop":.30, "TikTok":.55, "Saver":.12, "Betting":.45, "Gacha":.30}

# Market (Phase 1) -- calibrated from real index data (Max Sharpe, QQQ/SPY/GLD/TLT 2019-2024)
PORT_MU, PORT_SIG = 0.0110, 0.0417
VIX_SHOCK_MONTHS  = {2, 53}   # ~Mar2020 COVID, ~Aug2024 carry-trade
# Note: this standalone module uses suppress mode (×0.7) only.
# The notebook (GachaNomics_V6.ipynb) has a VIX_MODE toggle for robustness.

# Activation threshold sweep (reported for sensitivity)
TAU_SWEEP = [0.10, 0.15, 0.20]
THRESHOLD = 0.15

# LLM calibration toggle
# Set to True + export ANTHROPIC_API_KEY to enable one-time LLM damage validation.
# False (default): runs entirely in numpy, ~5 seconds, no API key required.
USE_LLM_CALIBRATION = False
LLM_MODEL = "claude-haiku-4-5"
LLM_SLEEP = 12

# ============================================================================
# PHASE 1 -- MARKET RETURNS
# ============================================================================
def build_market_returns(seed=SEED):
    rng = np.random.RandomState(seed)
    returns = rng.normal(PORT_MU, PORT_SIG, N_MONTHS)
    ytd = np.zeros(N_MONTHS)
    for m in range(N_MONTHS):
        lo = max(0, m - 11)
        ytd[m] = np.prod(1 + returns[lo:m+1]) - 1
    return returns, ytd

# ============================================================================
# PHASE 2 -- EVENT VECTOR CALENDAR (V6 core; real-world sourced)
# ============================================================================
def _cmn(m): return (m % 12) + 1
def _is_wc(m):
    return (_cmn(m) in (6, 7)) and ((m // 12) % 2 == 0)

def build_event_calendar():
    """60x7. Triggers ADD when they share a month -- the 'Perfect Storm' stack."""
    E = np.zeros((N_MONTHS, 7))
    for m in range(N_MONTHS):
        c = _cmn(m)
        E[m, GACHA]  += 2.0                                   # ~6-wk patch, 3-wk phases
        E[m, TIKTOK] += 2.5 if c in (9,10,11,12) else 0.5     # mega-sales vs payday
        E[m, MU]     += 1.0                                   # lottery 1st & 16th
        if c in (1, 4):      E[m, MU]  += 1.0                  # CNY, Songkran
        if c in (1,2,3,4,5,8,9,10,11,12): E[m, BET] += 1.5    # EPL Aug-May
        if _is_wc(m):        E[m, BET] += 2.5                 # World Cup/Euro mega
        if c in (5,10,11,12): E[m, COMP] += 1.5               # MSI/Worlds/ESL
        if m in VIX_SHOCK_MONTHS:                              # risk-off
            E[m, SAVER] += 2.0
            E[m, :6]    *= 0.7
    return E

# ============================================================================
# CALIBRATION -- per-axis beta from published Thai Gen Z baseline impulse rates
# ============================================================================
def calibrate_beta(EVENT):
    """beta_axis[k] so a near-pure archetype on axis k fires ~ its published baseline impulse prob."""
    dom = {"Mu":MU,"K-pop":KPOP,"TikTok":TIKTOK,"Saver":SAVER,"Betting":BET,"Gacha":GACHA}
    beta = np.zeros(7)
    for name, wv in ARCHETYPE_WEIGHTS.items():
        score = (wv @ EVENT.T).mean()
        beta[dom[name]] = -np.log(1 - BASELINE_IMPULSE_P[name]) / max(score, 1e-6)
    beta[COMP] = beta[beta > 0].mean()   # no pure-Comp archetype; use mean
    return beta

# ============================================================================
# PHASE 3 -- OPTIONAL LLM VALIDATION OF BASE_DMG
# ============================================================================
def llm_calibrate_base_dmg(base_dmg):
    if not USE_LLM_CALIBRATION:
        print("[LLM] OFF -> using real-world BASE_DMG as-is."); return base_dmg
    try:
        import anthropic
    except ImportError:
        print("[LLM] SDK missing -> BASE_DMG as-is."); return base_dmg
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[LLM] no API key -> BASE_DMG as-is."); return base_dmg
    client = anthropic.Anthropic()
    ctx = {"Mu":"amulet/lottery ~B300-1000","K-Pop":"concert ticket B2500-8700",
           "TikTok":"mega-sale basket ~B14000","Gacha":"pull-spree B400-1000/mo",
           "Competitive":"skin/battlepass SEA ARPU ~$18.8","Betting":"online avg ~B1633/mo",
           "Saver":"none"}
    out = base_dmg.copy()
    sysp = ("You are a Thai behavioral-finance researcher. Given an anchor THB value and "
            "context, return ONLY {\"thb\": <number>} within +/-30% of the anchor.")
    for i, ax in enumerate(AXES):
        if base_dmg[i] == 0: continue
        try:
            msg = client.messages.create(model=LLM_MODEL, max_tokens=128, system=sysp,
                messages=[{"role":"user","content":
                    f"Axis:{ax}\nContext:{ctx[ax]}\nAnchor THB:{base_dmg[i]:.0f}\nJSON only."}])
            t = msg.content[0].text.strip().strip("`")
            if t.startswith("json"): t = t[4:].strip()
            v = float(json.loads(t)["thb"])
            out[i] = min(max(v, base_dmg[i]*0.7), base_dmg[i]*1.3)
            print(f"[LLM] {ax:12s} {base_dmg[i]:7.0f} -> {out[i]:7.0f}")
        except Exception as e:
            print(f"[LLM] {ax} failed ({e}) -> anchor"); out[i] = base_dmg[i]
        time.sleep(LLM_SLEEP)
    return out

# ============================================================================
# PHASE 4A -- VERSION A: DIRICHLET DNA
# ============================================================================
def generate_dna_population(n=N_AGENTS, seed=SEED):
    """n x 7, rows sum to 1.0. alpha = Thai marginal prevalence. Independent draw."""
    rng = np.random.RandomState(seed + 1)
    alpha = np.array([0.73, 0.27, 0.77, 0.55, 0.11, 0.40, 0.60])
    return rng.dirichlet(alpha, size=n)

def activation_A(DNA, EVENT, beta, threshold, seed=SEED+2):
    """Probabilistic firing + by-construction guardrail (equal total intensity)."""
    rng = np.random.RandomState(seed)
    eff_beta = (DNA @ beta) / np.maximum(DNA.sum(axis=1), 1e-9)     # n,
    score = DNA @ EVENT.T                                          # n x 60
    score = score / np.maximum(DNA.sum(axis=1, keepdims=True), 1e-9)  # GUARDRAIL
    score = np.maximum(score - threshold, 0.0)                     # threshold floor
    p = 1 - np.exp(-eff_beta[:, None] * score)
    fire = rng.random((DNA.shape[0], N_MONTHS)) < p
    base = DNA @ BASE_DMG                                          # n,
    spend = fire * base[:, None] * rng.lognormal(0, DMG_SIGMA, fire.shape)
    # whale tail (exposure-weighted real THB)
    wp = DNA @ WHALE_PROB; wthb = DNA @ WHALE_THB
    spend += (rng.random(fire.shape) < wp[:, None]) * wthb[:, None]
    return spend

# ============================================================================
# PHASE 4B -- VERSION B: 6 ARCHETYPES
# ============================================================================
ARCHETYPE_WEIGHTS = {
    "Mu":      np.array([.80,0,.15,0,0,0,.05]),
    "K-pop":   np.array([.10,.50,.20,0,0,0,.20]),
    "TikTok":  np.array([.20,0,.65,0,0,0,.15]),
    "Saver":   np.array([.05,0,.30,0,0,0,.65]),
    "Betting": np.array([.05,0,.10,0,0,.80,.05]),
    "Gacha":   np.array([.05,0,.15,.75,.05,0,0]),
}
ARCHETYPE_FIN = {
    "Mu":     dict(dca=2500, init=50000),  "K-pop":  dict(dca=4000, init=80000),
    "TikTok": dict(dca=5000, init=100000), "Saver":  dict(dca=8000, init=150000),
    "Betting":dict(dca=3000, init=60000),  "Gacha":  dict(dca=3500, init=70000),
}

def activation_B(wv, EVENT, beta, threshold, n_paths, seed):
    rng = np.random.RandomState(seed)
    eff_beta = (wv @ beta) / max(wv.sum(), 1e-9)
    score = np.maximum(wv @ EVENT.T - threshold, 0.0)             # 60,
    p = 1 - np.exp(-eff_beta * score)
    fire = rng.random((n_paths, N_MONTHS)) < p[None, :]
    base = float(wv @ BASE_DMG)
    spend = fire * base * rng.lognormal(0, DMG_SIGMA, (n_paths, N_MONTHS))
    wp = float(wv @ WHALE_PROB); wthb = float(wv @ WHALE_THB)
    spend += (rng.random((n_paths, N_MONTHS)) < wp) * wthb
    return spend

# ============================================================================
# LOSS-CHASE (Betting only) -- separate from lambda
# ============================================================================
def apply_loss_chase(spend, betting_share, returns):
    spend = spend.copy()
    loss = returns < 0
    bs = np.asarray(betting_share)
    for m in range(1, N_MONTHS):
        if loss[m-1]:
            mult = 1.0 + (LOSS_CHASE_MULT - 1.0) * bs
            spend[:, m] *= mult if bs.ndim == 0 else mult
    return spend

# ============================================================================
# PHASE 5 -- SHARED SINGLE-POOL WEALTH/DEBT ENGINE
# ============================================================================
def simulate_wealth(spend, returns, dca, init_wealth):
    k = spend.shape[0]
    dca = np.full(k, dca, float) if np.isscalar(dca) else dca.astype(float)
    w = (np.full(k, init_wealth, float) if np.isscalar(init_wealth) else init_wealth.astype(float))
    debt = np.zeros(k); ideal = w.copy()
    for m in range(N_MONTHS):
        r = returns[m]
        w *= (1+r); ideal *= (1+r); ideal += dca
        serv = np.minimum(np.maximum(debt*MIN_PAYMENT_RATE, np.where(debt>0,300.,0.)), debt)
        w += np.maximum(dca - serv, 0.); debt -= serv
        s = spend[:, m]; pay = np.minimum(s, np.maximum(w, 0)); w -= pay; debt += s - pay
        debt *= (1 + BNPL_RATE)
    spiral = debt > (DEBT_SPIRAL_K * dca)
    return w, debt, spiral, ideal

# ============================================================================
# RUNNERS
# ============================================================================
AXIS_INIT = np.array([50000,80000,100000,70000,70000,60000,150000.])
AXIS_DCA  = np.array([2500,4000,5000,3500,3500,3000,8000.])

def run_version_A(EVENT, returns, beta, threshold):
    DNA = generate_dna_population()
    spend = activation_A(DNA, EVENT, beta, threshold)
    spend = apply_loss_chase(spend, DNA[:, BET], returns)
    init_w = DNA @ AXIS_INIT; dca_w = DNA @ AXIS_DCA
    w, d, sp, ideal = simulate_wealth(spend, returns, dca_w, init_w)
    gap = (ideal - w) / np.maximum(ideal, 1)
    true_hybrid = (np.sum(DNA > 0.05, axis=1) >= 4).mean() * 100
    return dict(meanW=w.mean(), meanD=d.mean(), spiral=sp.mean()*100,
                gap=gap.mean()*100, Wstd=w.std(), hybrid=true_hybrid)

def run_version_B(EVENT, returns, beta, threshold):
    rows = {}
    for i, (name, wv) in enumerate(ARCHETYPE_WEIGHTS.items()):
        fin = ARCHETYPE_FIN[name]
        spend = activation_B(wv, EVENT, beta, threshold, N_PATHS_B, seed=SEED+10+i)
        spend = apply_loss_chase(spend, float(wv[BET]), returns)
        w, d, sp, ideal = simulate_wealth(spend, returns, fin["dca"], fin["init"])
        gap = (ideal - w) / np.maximum(ideal, 1)
        rows[name] = dict(meanW=w.mean(), meanD=d.mean(),
                          spiral=sp.mean()*100, gap=gap.mean()*100)
    return rows

# ============================================================================
# MAIN (numbers-only)
# ============================================================================
def main():
    print("="*74)
    print("THE GACHA-NOMICS V6 -- Context-Activated DNA (numbers-only)")
    print("="*74)
    returns, ytd = build_market_returns()
    EVENT = build_event_calendar()
    global BASE_DMG
    BASE_DMG = llm_calibrate_base_dmg(BASE_DMG)
    beta = calibrate_beta(EVENT)

    print(f"\nMarket: mean {returns.mean()*100:.2f}%/mo  vol {returns.std()*100:.2f}%  "
          f"VIX-shock months {sorted(VIX_SHOCK_MONTHS)}")
    stk = EVENT.sum(axis=1)
    print(f"Event calendar 60x7 | Perfect-Storm max stack {stk.max():.1f} @ month {stk.argmax()}")
    print("per-axis beta:", ", ".join(f"{a}={b:.2f}" for a,b in
          zip(["Mu","Kp","Tk","Ga","Co","Be","Sa"], beta)))

    for tau in TAU_SWEEP:
        tag = "  <-- baseline" if tau == THRESHOLD else ""
        print("\n" + "-"*74); print(f"THRESHOLD tau = {tau}{tag}"); print("-"*74)
        B = run_version_B(EVENT, returns, beta, tau)
        print(f"{'VERSION B':<13}{'fire-cal':>9}{'FinalW':>11}{'Debt':>10}{'Spiral%':>9}{'Gap%':>7}")
        for name, r in B.items():
            print(f"  {name:<11}{BASELINE_IMPULSE_P[name]*100:>7.0f}%{r['meanW']:>11,.0f}"
                  f"{r['meanD']:>10,.0f}{r['spiral']:>8.1f}%{r['gap']:>6.1f}%")
        A = run_version_A(EVENT, returns, beta, tau)
        print(f"\n  VERSION A (10k hybrids): meanW {A['meanW']:,.0f} | debt {A['meanD']:,.0f} "
              f"| spiral {A['spiral']:.1f}% | gap {A['gap']:.1f}% | Wstd {A['Wstd']:,.0f}")
        print(f"  H6: {A['hybrid']:.1f}% are true hybrids (>=4 axes active >5%)")

    print("\n" + "="*74)
    print("LABELS: DNA independent (co-occurrence=survey future work). Single-pool")
    print("engine (single-pool; non-fungible bucket structure is future work). Gacha=resetting")
    print("pity only. Spiral=emergent from real whale magnitudes, NOT fitted to any pre-specified target.")
    print("Betting fragility=loss-chasing mult, not lambda.")
    print("="*74)

if __name__ == "__main__":
    main()
