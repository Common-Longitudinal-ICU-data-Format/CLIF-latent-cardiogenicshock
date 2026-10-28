#!/usr/bin/env python3

"""
tests for the sensitivity smoother

The important test here is RECOVERY FROM SIMULATION: a Kalman smoother will
return a smooth, plausible curve from pure noise, so the only way to know it
estimates anything is to give it data with a known answer and check it finds
it. Two of these tests would pass on a broken implementation that always
returned the prior, which is why the third one exists.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))

from clifcs.sensitivity import (  # noqa: E402
    saturating, smooth_one, unobserved_variance,
)

PRIOR_MEAN = [70.0, 0.0]
PRIOR_VAR = [400.0, 1.0]


class TestSaturating:
    def test_log_is_zero_at_zero_dose(self):
        assert saturating([0.0], "log")[0] == 0.0

    def test_emax_is_half_at_ed50(self):
        assert saturating([0.1], "emax", ed50=0.1)[0] == pytest.approx(0.5)

    def test_both_saturate(self):
        # The SAME additive step in dose must buy less at a higher dose. An
        # earlier version of this test compared a step of 0.05 against a step
        # of 0.8 and failed for that reason rather than because the functions
        # were wrong -- saturation is about the derivative, so the steps have
        # to be equal.
        for kind in ("log", "emax"):
            low = saturating([0.1, 0.2], kind)
            high = saturating([1.0, 1.1], kind)
            assert (low[1] - low[0]) > (high[1] - high[0])

    def test_negative_dose_is_clamped(self):
        assert saturating([-1.0], "log")[0] == 0.0

    def test_unknown_form_raises(self):
        with pytest.raises(ValueError):
            saturating([0.1], "linear")


class TestSmoother:
    def test_recovers_a_known_constant_slope(self):
        # the test that can actually fail: simulate with a true s and check it
        rng = np.random.default_rng(0)
        n = 300
        dose = np.abs(rng.normal(0.3, 0.2, n))
        regressor = saturating(dose, "log")
        true_s, true_baseline = 8.0, 72.0
        y = true_baseline + true_s * regressor + rng.normal(0, 2.0, n)
        state, _cov = smooth_one(
            y, regressor, obs_var=4.0, walk_var=np.array([0.01, 0.0005]),
            prior_mean=PRIOR_MEAN, prior_var=PRIOR_VAR,
        )
        assert state[n // 2, 1] == pytest.approx(true_s, abs=1.5)
        assert state[n // 2, 0] == pytest.approx(true_baseline, abs=2.0)

    def test_returns_the_prior_when_dose_never_moves(self):
        # with a constant regressor the slope is not separable from baseline,
        # so the estimate must stay near its prior rather than invent a value
        n = 200
        regressor = np.full(n, 0.5)
        y = np.full(n, 70.0)
        state, cov = smooth_one(
            y, regressor, obs_var=4.0, walk_var=np.array([0.01, 0.0005]),
            prior_mean=PRIOR_MEAN, prior_var=PRIOR_VAR,
        )
        assert abs(state[n // 2, 1]) < 1.0

    def test_missing_observations_widen_the_band(self):
        rng = np.random.default_rng(1)
        n = 120
        dose = np.abs(rng.normal(0.3, 0.2, n))
        regressor = saturating(dose, "log")
        y = 70.0 + 5.0 * regressor + rng.normal(0, 2.0, n)
        gapped = y.copy()
        gapped[40:100] = np.nan
        _s_full, cov_full = smooth_one(
            y, regressor, 4.0, np.array([0.01, 0.0005]), PRIOR_MEAN, PRIOR_VAR
        )
        _s_gap, cov_gap = smooth_one(
            gapped, regressor, 4.0, np.array([0.01, 0.0005]),
            PRIOR_MEAN, PRIOR_VAR
        )
        assert cov_gap[70, 1, 1] > cov_full[70, 1, 1]

    def test_all_missing_leaves_the_prior_untouched(self):
        n = 50
        state, _cov = smooth_one(
            np.full(n, np.nan), np.full(n, 0.5), 4.0,
            np.array([0.01, 0.0005]), PRIOR_MEAN, PRIOR_VAR
        )
        assert state[:, 1] == pytest.approx(np.zeros(n), abs=1e-9)

    def test_unobserved_variance_grows_with_time(self):
        grown = unobserved_variance(10, np.array([0.01, 0.05]), PRIOR_VAR)
        assert grown[0] == pytest.approx(1.0)
        assert grown[9] > grown[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
