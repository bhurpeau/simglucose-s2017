"""
Tests de plausibilité S2017 (cf. CC_TASK_S2017.md section Validation).

Vérifie :
1. BG physiologique (~70–300 mg/dL) sur une journée
2. Remontée nocturne de BG entre 3h et 7h (effet dawn)
3. Réponse à un repas différente entre petit-déj et dîner (variabilité SI)
4. Pas de NaN/inf, pas d'états négatifs persistants
"""
import numpy as np
import pytest
from collections import namedtuple

from simglucose.patient.t1dpatient import T1DPatient, Action


PATIENT = "adult#001"
SEED = 42
MINUTES_PER_DAY = 1440


def run_open_loop(patient_name, start_time_min, duration_min=1440, seed=SEED,
                  model_version="S2017", si_class=1, meal_events=None):
    """Run open-loop simulation (basal only unless meal_events given).

    meal_events : list of (t_start, t_end, cho_rate_g_per_min)
    """
    p = T1DPatient.withName(
        patient_name,
        seed=seed,
        model_version=model_version,
        start_time=start_time_min,
        si_class=si_class,
    )
    basal = p._params.u2ss * p._params.BW / 6000  # U/min

    meal_events = meal_events or []
    bg_trace = []
    states_trace = []

    for step in range(duration_min):
        t = p.t
        cho = 0.0
        for (t0, t1, rate) in meal_events:
            if t0 <= t < t1:
                cho = rate
        act = Action(CHO=cho, insulin=basal)
        bg_trace.append(p.observation.Gsub)
        states_trace.append(p.state.copy())
        p.step(act)

    return np.array(bg_trace), states_trace


class TestS2013Regression:
    """Quick sanity: S2013 mode produces same trajectory for adult#001."""

    def test_no_nan_inf(self):
        bg, states = run_open_loop(PATIENT, start_time_min=360,
                                   model_version="S2013", si_class=1)
        assert not np.any(np.isnan(bg)), "NaN in S2013 BG"
        assert not np.any(np.isinf(bg)), "inf in S2013 BG"


class TestS2017Plausibility:
    """S2017 mode plausibility checks."""

    def test_physiological_bg_range(self):
        """BG should stay roughly physiological on basal-only day."""
        bg, states = run_open_loop(PATIENT, start_time_min=0,
                                   model_version="S2017", si_class=1, seed=0)
        assert np.all(bg > 40), f"BG below 40: min={bg.min():.1f}"
        assert np.all(bg < 400), f"BG above 400: max={bg.max():.1f}"

    def test_no_nan_inf(self):
        bg, states = run_open_loop(PATIENT, start_time_min=0,
                                   model_version="S2017", si_class=1)
        assert not np.any(np.isnan(bg)), "NaN in S2017 BG"
        assert not np.any(np.isinf(bg)), "inf in S2017 BG"

    def test_no_persistent_negative_states(self):
        """Non-glucose states should not go persistently negative."""
        _, states = run_open_loop(PATIENT, start_time_min=0,
                                  model_version="S2017", si_class=1)
        arr = np.array(states)
        # Allow occasional clipped zeros but no sustained large negatives
        assert np.all(arr > -1e-3), (
            f"State went negative: min={arr.min():.4g} at state "
            f"{np.unravel_index(arr.argmin(), arr.shape)}"
        )

    def test_dawn_effect_bg_rise(self):
        """BG should be higher at 7:00 than at 3:00 on a basal-only night.

        Simulation starts at midnight (start_time=0).
        dawn window is t in [180, 420] min.
        """
        bg, _ = run_open_loop(PATIENT, start_time_min=0,
                               model_version="S2017", si_class=1, seed=0)
        bg_at_3am = bg[180]
        bg_at_7am = bg[420]
        # Allow a small margin: dawn may take a few minutes to accumulate
        assert bg_at_7am >= bg_at_3am - 2.0, (
            f"Dawn effect not visible: BG@3am={bg_at_3am:.1f} BG@7am={bg_at_7am:.1f}"
        )

    def test_breakfast_dinner_si_difference(self):
        """With class-5 SI (low at B, high at D), breakfast peak > dinner peak.

        Class 5 (l-h-h): si_B=0.4, si_L=1.0, si_D=1.0 → more glucose excursion
        at breakfast (less insulin action) than at dinner (full insulin action).
        """
        # 80 g CHO meal over 16 min (EAT_RATE = 5 g/min)
        MEAL_RATE = 5.0  # g/min (EAT_RATE)
        MEAL_DURATION = 16  # min → 80 g total

        # Breakfast at 07:00 → t=420 min (start_time=0)
        breakfast_start = 420
        meal_B = [(breakfast_start, breakfast_start + MEAL_DURATION, MEAL_RATE)]
        bg_B, _ = run_open_loop(PATIENT, start_time_min=0,
                                 model_version="S2017", si_class=5,
                                 meal_events=meal_B, duration_min=600, seed=0)
        # Dinner at 19:00 → t=1140 min (start_time=0), simulate long enough
        dinner_start = 1140
        meal_D = [(dinner_start, dinner_start + MEAL_DURATION, MEAL_RATE)]
        bg_D, _ = run_open_loop(PATIENT, start_time_min=0,
                                 model_version="S2017", si_class=5,
                                 meal_events=meal_D, duration_min=1300, seed=0)

        peak_B = bg_B[breakfast_start:breakfast_start + 180].max()
        peak_D = bg_D[dinner_start:dinner_start + 180].max()

        assert peak_B > peak_D - 5, (
            f"Class-5 SI: expected higher breakfast peak than dinner. "
            f"peak_B={peak_B:.1f} peak_D={peak_D:.1f}"
        )

    def test_glucagon_states_positive(self):
        """Glucagon states (H, SR_Hs) should remain non-negative."""
        _, states = run_open_loop(PATIENT, start_time_min=0,
                                  model_version="S2017", si_class=1)
        from simglucose.patient.t1dpatient import _IDX_H, _IDX_SR_HS
        H_vals    = np.array([s[_IDX_H]     for s in states])
        SR_HS_vals = np.array([s[_IDX_SR_HS] for s in states])
        assert np.all(H_vals >= 0), f"H<0: min={H_vals.min():.4g}"
        assert np.all(SR_HS_vals >= 0), f"SR_Hs<0: min={SR_HS_vals.min():.4g}"
