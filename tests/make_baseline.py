"""
Génère la trace de référence S2013 AVANT toute modification S2017.
Sert de test de non-régression : après les modifs, une simulation en mode
S2013 (flag par défaut) doit reproduire ce CSV à l'identique.

Usage :
    python tests/make_baseline.py        # (re)génère tests/baseline_s2013.csv
    python tests/make_baseline.py --check # compare la sim actuelle au baseline
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.simulation.sim_engine import SimObj, sim

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline_s2013.csv")

# Scénario de référence FIXE — ne pas modifier, sinon le baseline n'a plus de sens.
PATIENT = "adult#001"
START = datetime(2024, 1, 1, 6, 0, 0)
MEALS = [(1.0, 45), (7.0, 70), (13.0, 80)]  # (h depuis start, g CHO)
SEED = 1
DAYS = 1


def run():
    env = T1DSimEnv(
        patient=T1DPatient.withName(PATIENT),
        sensor=CGMSensor.withName("Dexcom", seed=SEED),
        pump=InsulinPump.withName("Insulet"),
        scenario=CustomScenario(start_time=START, scenario=MEALS),
    )
    sim_obj = SimObj(env, BBController(), timedelta(days=DAYS),
                     animate=False, path=os.path.join(HERE, "_tmp_results"))
    return sim(sim_obj)[["BG", "CGM", "CHO", "insulin"]]


def main():
    res = run()
    if "--check" in sys.argv:
        if not os.path.exists(BASELINE):
            sys.exit("Pas de baseline. Lance d'abord sans --check.")
        ref = pd.read_csv(BASELINE, index_col=0)
        # tolérance numérique serrée : la dynamique doit être identique en mode S2013
        ok = np.allclose(res["BG"].values, ref["BG"].values, rtol=0, atol=1e-6)
        if ok:
            print("OK — non-régression S2013 respectée.")
        else:
            diff = np.abs(res["BG"].values - ref["BG"].values)
            sys.exit(f"REGRESSION — écart max BG = {diff.max():.4g} mg/dL")
    else:
        res.to_csv(BASELINE)
        print(f"Baseline écrit : {BASELINE} ({len(res)} pas)")
        print(f"BG min/max : {res.BG.min():.1f} / {res.BG.max():.1f} mg/dL")


if __name__ == "__main__":
    main()
