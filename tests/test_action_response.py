#!/usr/bin/env python3

"""
unit tests for the action-response state encoding

These test the CLASSIFICATION RULES, not the data. Every case below is a
hand-written dose and MAP pair whose correct label can be read off the
definition, which is what makes them able to fail: a test built by running the
encoder over real patients and freezing its output only proves the encoder
still does what it did, including whatever it got wrong.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))

from clifcs.action_response import (  # noqa: E402
    UNDECIDABLE, classify_action, classify_state,
)

RELATIVE, ABSOLUTE = 0.25, 0.01
ADEQUATE, DEADBAND = 65.0, 3.0


def action(start, end):
    return classify_action(start, end, RELATIVE, ABSOLUTE)


class TestAction:
    def test_both_floors_must_be_cleared(self):
        # +50% relative but only +0.005 absolute: too small to be a decision
        assert action(0.01, 0.015) == "no change"
        # +0.10 absolute but only +8% relative on a big dose
        assert action(1.20, 1.30) == "no change"
        # clears both
        assert action(0.10, 0.20) == "escalate"

    def test_de_escalation_is_symmetric(self):
        assert action(0.20, 0.10) == "de-escalate"
        assert action(0.015, 0.01) == "no change"

    def test_starting_from_zero_uses_the_absolute_floor_only(self):
        # a relative change from zero is undefined, not infinite
        assert action(0.0, 0.05) == "escalate"
        assert action(0.0, 0.005) == "no change"

    def test_stopping_to_zero_is_a_de_escalation(self):
        assert action(0.20, 0.0) == "de-escalate"

    def test_missing_dose_is_not_an_action(self):
        assert action(None, 0.1) is None
        assert action(0.1, None) is None


class TestState:
    def test_escalation_splits_on_the_map_deadband(self):
        assert classify_state("escalate", 60.0, 5.0, False, ADEQUATE,
                              DEADBAND) == "responsive escalation"
        # inside the band is not a response
        assert classify_state("escalate", 60.0, 2.0, False, ADEQUATE,
                              DEADBAND) == "refractory escalation"
        assert classify_state("escalate", 60.0, -4.0, False, ADEQUATE,
                              DEADBAND) == "refractory escalation"

    def test_wean_fails_when_the_dose_goes_back_up(self):
        # MAP held, but the clinician reversed it: that is the clinician's own
        # verdict and it counts as a failure
        assert classify_state("de-escalate", 70.0, 0.0, True, ADEQUATE,
                              DEADBAND) == "failed wean"

    def test_wean_fails_when_the_pressure_falls(self):
        assert classify_state("de-escalate", 70.0, -6.0, False, ADEQUATE,
                              DEADBAND) == "failed wean"

    def test_wean_tolerated_only_if_map_stays_adequate(self):
        assert classify_state("de-escalate", 70.0, 0.0, False, ADEQUATE,
                              DEADBAND) == "tolerated wean"
        # held steady but below 65 throughout: not tolerated
        assert classify_state("de-escalate", 60.0, 0.0, False, ADEQUATE,
                              DEADBAND) == "failed wean"

    def test_hold_splits_on_drift(self):
        assert classify_state("no change", 70.0, 0.0, False, ADEQUATE,
                              DEADBAND) == "stable hold"
        assert classify_state("no change", 70.0, -8.0, False, ADEQUATE,
                              DEADBAND) == "drifting hold"
        # a hold that drifts UP is still stable, not a response to anything
        assert classify_state("no change", 70.0, 8.0, False, ADEQUATE,
                              DEADBAND) == "stable hold"

    def test_off_pressors_uses_level_not_change(self):
        assert classify_state("off", 70.0, -9.0, False, ADEQUATE,
                              DEADBAND) == "off pressors, MAP >=65"
        assert classify_state("off", 60.0, 9.0, False, ADEQUATE,
                              DEADBAND) == "off pressors, MAP <65"

    def test_the_adequate_threshold_is_inclusive(self):
        assert classify_state("off", 65.0, 0.0, False, ADEQUATE,
                              DEADBAND) == "off pressors, MAP >=65"

    def test_missing_response_is_undecidable_not_guessed(self):
        assert classify_state("escalate", 60.0, None, False, ADEQUATE,
                              DEADBAND) == UNDECIDABLE
        assert classify_state("no change", None, 1.0, False, ADEQUATE,
                              DEADBAND) == UNDECIDABLE
        # but an off-pressor step needs only the level
        assert classify_state("off", 70.0, None, False, ADEQUATE,
                              DEADBAND) == "off pressors, MAP >=65"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
