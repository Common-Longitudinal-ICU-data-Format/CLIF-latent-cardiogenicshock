#!/usr/bin/env python3

"""
how much the blood pressure moves per unit of vasoactive dose, over time

Two models of the same thing, fitted because the data supports one of them much
better than the other and the comparison is the result.

  GAIN      MAP(t) = baseline(t) + s(t) * f(NEE(t)) + noise, with baseline and
            s as random walks. This is the specified model. With f(NEE) an
            observed regressor it is a linear-Gaussian state-space model whose
            observation matrix is [1, f(NEE_t)], so a Kalman smoother gives
            s(t) with a posterior variance at every hour.

  DIRECTION a time-varying probability that the pressure rises after an
            escalation (or holds after a wean), estimated by exponentially
            weighted beta-binomial shrinkage toward the cohort rate.

WHY BOTH. In this cohort the correlation between dose and MAP is ~0 at every
formulation tried -- -0.002 in levels, -0.000 in differences, and -0.004 to
+0.003 across lags of one to six hours. Meanwhile the conditional means
separate cleanly: MAP rises 1.8 mmHg in the hour after an escalation and falls
1.4 after a wean, widening to +2.8 and -1.5 by six hours. The DIRECTION of a
dose change predicts the direction of the response; the MAGNITUDE predicts
nothing. That is what titrating to a target plus confounding by severity
produces -- a big escalation happens in a sicker patient whose pressure
responds less, so magnitude effects cancel while sign survives.

So the gain model is close to unidentified here, and a random walk with partial
pooling will still return a smooth, plausible s(t) made almost entirely of
shrinkage. `data_share` is computed for exactly that reason: it is the fraction
of each patient's posterior precision on s that came from their own
observations rather than from the prior, and a sensitivity trajectory with a
low data_share should not be stratified on.

NONE OF THIS IS A DOSE EFFECT. Doses are not assigned. A patient whose pressure
rises after an escalation may have been recovering anyway, and the clinician
escalated because it was falling. The quantity estimated is an association
between a recorded dose change and a subsequent recorded pressure change.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def saturating(nee, kind: str, reference: float = 0.1, ed50: float = 0.1):
    """
    the dose put on a saturating scale

    Both forms are bounded in the dose and roughly linear at low dose, which is
    what makes them saturating rather than merely transformed. `reference` and
    `ed50` set where the curve bends; they are POOLED rather than fitted per
    patient, because a per-patient bend point cannot be identified when the
    per-patient gain cannot be.
    """
    dose = np.asarray(nee, dtype=np.float64)
    if kind == "log":
        return np.log1p(np.maximum(dose, 0.0) / reference)
    if kind == "emax":
        return np.maximum(dose, 0.0) / (np.maximum(dose, 0.0) + ed50)
    raise ValueError(f"unknown saturating form {kind!r}")


def smooth_one(y, regressor, obs_var, walk_var, prior_mean, prior_var):
    """
    Kalman filter and RTS smoother for one patient's [baseline, s] pair

    The state is two-dimensional and the observation matrix is time-varying,
    H_t = [1, f(NEE_t)], which is what lets the same recursion estimate a drift
    in the untreated pressure and a drift in the dose response at once.

    MISSING OBSERVATIONS ARE PREDICTED THROUGH, not skipped: an hour with no
    MAP advances the state and inflates its variance, which is the correct
    behaviour for a random walk and is why a patient with sparse monitoring
    gets wider bands rather than a shorter series.

    Returns the smoothed mean and variance of s at every hour, plus the share
    of final posterior precision on s that came from the data. That share is
    1 - (posterior variance / prior variance at the same hour under no
    observations), computed by running the same recursion with every
    observation masked, so it answers "how much did this patient's own data
    move the estimate" rather than "is the number plausible".
    """
    n = len(y)
    walk = np.diag(walk_var)
    state = np.array(prior_mean, dtype=np.float64)
    covariance = np.diag(prior_var).astype(np.float64)

    predicted_state = np.zeros((n, 2))
    predicted_cov = np.zeros((n, 2, 2))
    filtered_state = np.zeros((n, 2))
    filtered_cov = np.zeros((n, 2, 2))

    for t in range(n):
        if t:
            state = filtered_state[t - 1].copy()
            covariance = filtered_cov[t - 1] + walk
        predicted_state[t] = state
        predicted_cov[t] = covariance
        if np.isnan(y[t]):
            filtered_state[t] = state
            filtered_cov[t] = covariance
            continue
        design = np.array([1.0, regressor[t]])
        innovation_var = float(design @ covariance @ design) + obs_var
        gain = (covariance @ design) / innovation_var
        residual = y[t] - float(design @ state)
        filtered_state[t] = state + gain * residual
        filtered_cov[t] = covariance - np.outer(gain, design @ covariance)

    smoothed_state = filtered_state.copy()
    smoothed_cov = filtered_cov.copy()
    for t in range(n - 2, -1, -1):
        try:
            smoother = filtered_cov[t] @ np.linalg.inv(predicted_cov[t + 1])
        except np.linalg.LinAlgError:
            continue
        smoothed_state[t] = filtered_state[t] + smoother @ (
            smoothed_state[t + 1] - predicted_state[t + 1]
        )
        smoothed_cov[t] = filtered_cov[t] + smoother @ (
            smoothed_cov[t + 1] - predicted_cov[t + 1]
        ) @ smoother.T
    return smoothed_state, smoothed_cov


def unobserved_variance(n, walk_var, prior_var):
    """
    what the posterior variance of s would be with no observations at all

    The denominator for `data_share`. A random walk's variance grows linearly
    with time, so this is not a constant and comparing a patient's posterior
    variance against the prior at hour zero would make every long series look
    informative.
    """
    return prior_var[1] + walk_var[1] * np.arange(n, dtype=np.float64)


def pooled_loglik(params, series, prior_mean, prior_var):
    """
    negative log likelihood of the hyperparameters over a sample of patients

    The three variances are shared across patients -- that is where the partial
    pooling enters this fit. Parameterised on the log scale so the optimiser
    cannot step negative.
    """
    obs_var, baseline_walk, slope_walk = np.exp(params)
    walk_var = np.array([baseline_walk, slope_walk])
    total = 0.0
    for y, regressor in series:
        n = len(y)
        state = np.array(prior_mean, dtype=np.float64)
        covariance = np.diag(prior_var).astype(np.float64)
        for t in range(n):
            if t:
                covariance = covariance + np.diag(walk_var)
            if np.isnan(y[t]):
                continue
            design = np.array([1.0, regressor[t]])
            innovation_var = float(design @ covariance @ design) + obs_var
            residual = y[t] - float(design @ state)
            total += 0.5 * (
                np.log(2 * np.pi * innovation_var)
                + residual ** 2 / innovation_var
            )
            gain = (covariance @ design) / innovation_var
            state = state + gain * residual
            covariance = covariance - np.outer(gain, design @ covariance)
    return total


def fit_hyperparameters(series, prior_mean, prior_var, logger, seed=42,
                        sample=400):
    """
    the three shared variances, by maximum likelihood on a patient subsample

    EMPIRICAL BAYES, NOT FULL BAYES, and the difference is worth stating: the
    hyperparameters are estimated once and then treated as known, so the
    uncertainty in s(t) below does not include uncertainty in the variances
    that produced it. The bands are therefore slightly too narrow. A full
    hierarchical fit would propagate it, at a cost of hours rather than
    minutes over ten thousand patients.

    Fitted on a subsample because the likelihood is a sum over patients and
    several hundred series already pin three parameters tightly; the subsample
    is what keeps this a minutes-long step.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    if len(series) > sample:
        picked = rng.choice(len(series), size=sample, replace=False)
        series = [series[i] for i in picked]
    start = np.log([100.0, 1.0, 0.05])
    result = minimize(
        pooled_loglik, start, args=(series, prior_mean, prior_var),
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2},
    )
    obs_var, baseline_walk, slope_walk = np.exp(result.x)
    logger.info(
        f"  hyperparameters on {len(series):,} series: observation sd "
        f"{np.sqrt(obs_var):.2f} mmHg, baseline walk sd "
        f"{np.sqrt(baseline_walk):.3f} mmHg/h, slope walk sd "
        f"{np.sqrt(slope_walk):.4f} per hour"
        + ("" if result.success else "  [optimiser did not converge]")
    )
    return obs_var, np.array([baseline_walk, slope_walk])


def direction_probability(events, logger, forgetting: float = 0.97,
                          prior_weight: float = 8.0):
    """
    a time-varying probability that the pressure moves the way the dose did

    `events` is one row per titration with hospitalization_id, offset_hours and
    a boolean `went_the_right_way`. The estimate at hour t is a beta posterior
    whose counts are the patient's own past events discounted geometrically by
    `forgetting` per hour, plus a prior centred on the cohort rate and worth
    `prior_weight` events.

    NOT A KALMAN FILTER, deliberately. The observation is Bernoulli, so a
    linear-Gaussian recursion would need a linearisation whose approximation
    error is largest exactly where the data is thinnest -- in the patients this
    estimate exists to be honest about. Exponentially weighted beta-binomial
    shrinkage has a closed-form posterior at every hour, degrades to the
    cohort rate when a patient has no events, and makes the amount of
    borrowing explicit in `effective_events`.

    IT IS ESTIMATED ONLY WHERE TITRATIONS HAPPEN. Between them it carries
    forward with the counts decaying, so a long gap pulls the estimate back
    toward the cohort rate rather than holding a stale value -- which is the
    behaviour wanted: not having titrated for two days is not evidence that
    the last response still describes the patient.
    """
    if events.is_empty():
        logger.warning("no titration events; direction model skipped")
        return pl.DataFrame()
    base = float(events["went_the_right_way"].mean())
    logger.info(
        f"  direction model: {events.height:,} titrations, cohort rate "
        f"{base:.1%}, forgetting {forgetting} per hour, prior worth "
        f"{prior_weight:.0f} events"
    )
    rows = []
    for (patient,), part in events.sort("offset_hours").group_by(
        ["hospitalization_id"], maintain_order=True
    ):
        hours = part["offset_hours"].to_list()
        outcomes = part["went_the_right_way"].to_list()
        successes = 0.0
        trials = 0.0
        previous = hours[0] if hours else 0.0
        for hour, outcome in zip(hours, outcomes):
            decay = forgetting ** max(hour - previous, 0.0)
            successes *= decay
            trials *= decay
            previous = hour
            a = prior_weight * base + successes
            b = prior_weight * (1 - base) + (trials - successes)
            mean = a / (a + b)
            rows.append({
                "hospitalization_id": patient,
                "offset_hours": hour,
                "p_right_way": mean,
                "p_right_way_sd": float(
                    np.sqrt(mean * (1 - mean) / (a + b + 1))
                ),
                "effective_events": float(trials),
                "went_the_right_way": bool(outcome),
            })
            successes += 1.0 if outcome else 0.0
            trials += 1.0
    return pl.DataFrame(rows)
