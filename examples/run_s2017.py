"""
Exemple de simulation en mode S2017.

Montre :
- La variabilité SI intra-journalière (kp3, Vmx time-varying)
- Le phénomène de l'aube (kp1, kir)
- Le sous-système glucagon (états X_H, H, SR_Hs)

Usage :
    python examples/run_s2017.py
    python examples/run_s2017.py --compare   # superpose S2013 vs S2017

Dépendances : matplotlib (optionnel pour les graphes)
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simglucose.patient.t1dpatient import T1DPatient, Action, _IDX_H, _IDX_X_H, _IDX_SR_HS


def run(model_version, si_class=5, seed=42, start_time=0, duration=1440):
    """
    Simulate adult#001 for one day (1440 min) with three meals.

    Meals : breakfast at 07:00 (80 g), lunch at 13:00 (70 g), dinner at 19:00 (80 g).
    All meals delivered at EAT_RATE = 5 g/min.
    Insulin : basal rate only (open-loop, for illustration).

    Parameters
    ----------
    model_version : "S2013" or "S2017"
    si_class : 1-7 (only used in S2017)
    seed : RNG seed
    start_time : minutes from midnight when simulation starts
    duration : simulation length in minutes
    """
    p = T1DPatient.withName(
        "adult#001",
        seed=seed,
        model_version=model_version,
        start_time=start_time,
        si_class=si_class,
    )
    basal = p._params.u2ss * p._params.BW / 6000  # U/min

    # Meal schedule (absolute simulation times based on start_time)
    MEAL_RATE = 5.0   # g/min
    MEAL_DUR  = 16    # min  → 80 g per meal
    meal_schedule = []
    for meal_hhmm in [7 * 60, 13 * 60, 19 * 60]:   # 07:00, 13:00, 19:00
        t_start = (meal_hhmm - start_time) % 1440
        meal_schedule.append((t_start, t_start + MEAL_DUR, MEAL_RATE))

    times, BG, insulin_trace, CHO_trace = [], [], [], []
    H_trace, XH_trace = [], []

    for _ in range(duration):
        t = p.t
        cho = sum(rate for (t0, t1, rate) in meal_schedule if t0 <= t < t1)
        act = Action(CHO=cho, insulin=basal)
        times.append(t)
        BG.append(p.observation.Gsub)
        insulin_trace.append(act.insulin)
        CHO_trace.append(cho)
        if model_version == "S2017":
            H_trace.append(p.state[_IDX_H])
            XH_trace.append(p.state[_IDX_X_H])
        p.step(act)

    return {
        "t": np.array(times),
        "BG": np.array(BG),
        "insulin": np.array(insulin_trace),
        "CHO": np.array(CHO_trace),
        "H": np.array(H_trace) if H_trace else None,
        "X_H": np.array(XH_trace) if XH_trace else None,
        "model_version": model_version,
        "si_class": si_class if model_version == "S2017" else None,
    }


def print_summary(res):
    print(f"\n=== {res['model_version']} (SI class {res['si_class']}) ===")
    print(f"  BG  min/mean/max : {res['BG'].min():.1f} / {res['BG'].mean():.1f} / {res['BG'].max():.1f} mg/dL")
    if res["H"] is not None:
        print(f"  H   min/mean/max : {res['H'].min():.4f} / {res['H'].mean():.4f} / {res['H'].max():.4f} pmol/L")
        print(f"  X_H min/mean/max : {res['X_H'].min():.4f} / {res['X_H'].mean():.4f} / {res['X_H'].max():.4f}")

    # Dawn effect check
    if len(res["BG"]) > 420:
        bg_3am = res["BG"][180]
        bg_7am = res["BG"][420]
        print(f"  Dawn (03:00→07:00): BG {bg_3am:.1f} → {bg_7am:.1f} mg/dL "
              f"({'↑ dawn effect' if bg_7am > bg_3am else 'no dawn effect'})")


def main():
    parser = argparse.ArgumentParser(description="Simulate T1D S2017 for one day")
    parser.add_argument("--compare", action="store_true",
                        help="Also run S2013 for comparison")
    parser.add_argument("--si-class", type=int, default=5,
                        help="SI variability class 1-7 (default: 5 = l-h-h)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Suppress matplotlib plots")
    args = parser.parse_args()

    # Start at midnight so dawn window [180, 420] falls inside simulation
    res_s2017 = run("S2017", si_class=args.si_class, seed=42, start_time=0)
    print_summary(res_s2017)

    results = [res_s2017]
    if args.compare:
        res_s2013 = run("S2013", seed=42, start_time=0)
        print_summary(res_s2013)
        results.append(res_s2013)

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            n_panels = 4 if res_s2017["H"] is not None else 2
            fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)

            colors = {"S2017": "steelblue", "S2013": "tomato"}
            for res in results:
                t_h = res["t"] / 60
                lbl = f"{res['model_version']}" + (
                    f" class {res['si_class']}" if res["si_class"] else ""
                )
                axes[0].plot(t_h, res["BG"], color=colors[res["model_version"]],
                             label=lbl)
                axes[1].plot(t_h, res["CHO"], color=colors[res["model_version"]],
                             alpha=0.8)

            axes[0].axvspan(3, 7, alpha=0.1, color="orange", label="dawn window")
            axes[0].set_ylabel("BG (mg/dL)")
            axes[0].axhline(70, color="gray", ls="--", lw=0.7)
            axes[0].axhline(180, color="gray", ls="--", lw=0.7)
            axes[0].legend(fontsize=8)
            axes[1].set_ylabel("CHO (g/min)")

            if res_s2017["H"] is not None:
                axes[2].plot(t_h, res_s2017["H"], color="green", label="H (plasma glucagon)")
                axes[2].set_ylabel("H (pmol/L)")
                axes[2].legend(fontsize=8)
                axes[3].plot(t_h, res_s2017["X_H"], color="purple", label="X_H (glucagon action)")
                axes[3].set_ylabel("X_H")
                axes[3].legend(fontsize=8)

            axes[-1].set_xlabel("Time (h)")
            fig.suptitle("UVA/Padova T1D S2017 — adult#001 (open-loop, basal only)")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\n(matplotlib not installed — skipping plots)")


if __name__ == "__main__":
    main()
