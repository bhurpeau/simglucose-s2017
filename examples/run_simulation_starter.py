"""
Environnement de simulation T1D (UVA/Padova, version S2008/2013) via simglucose.
Point de depart pour un environnement de simulation.

Installation :
    pip install simglucose

Sujets disponibles dans la version PyPI (sous-ensemble public, 30 sujets) :
    adult#001 .. adult#010
    adolescent#001 .. adolescent#010
    child#001 .. child#010

Deux modes ci-dessous :
    A) boucle de simulation directe avec un controleur custom
    B) interface Gymnasium (RL-ready) -- gardee pour plus tard
"""

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# MODE A : boucle de simulation directe
# ---------------------------------------------------------------------------
# Convient quand tu veux juste un environnement deterministe ou tu pilotes
# toi-meme l'insuline et tu observes glucose/CGM. Tu controles tout le pas
# de temps.

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Controller, Action


class MyController(Controller):
    """Controleur minimal : basal constant, pas de bolus.
    Remplace policy() par ta propre logique. info['patient_state']
    te donne acces a l'etat complet du patient (13 etats du modele ODE)."""

    def __init__(self, init_state=None):
        self.init_state = init_state
        self.state = init_state

    def policy(self, observation, reward, done, **info):
        # observation.CGM = mesure du capteur (mg/dL)
        # info['patient_state'] = vecteur d'etat ODE complet
        basal = 0.0  # U/min  -- a remplacer
        bolus = 0.0  # U/min
        return Action(basal=basal, bolus=bolus)

    def reset(self):
        self.state = self.init_state


def run_direct(patient_name="adult#001", days=1, seed=1):
    start = datetime(2024, 1, 1, 6, 0, 0)
    # scenario repas : (heures_depuis_start, grammes_CHO)
    meals = [(1.0, 45), (7.0, 70), (13.0, 80)]

    env = T1DSimEnv(
        patient=T1DPatient.withName(patient_name),
        sensor=CGMSensor.withName("Dexcom", seed=seed),
        pump=InsulinPump.withName("Insulet"),
        scenario=CustomScenario(start_time=start, scenario=meals),
    )

    from simglucose.simulation.sim_engine import SimObj, sim

    sim_obj = SimObj(
        env,
        MyController(),
        timedelta(days=days),
        animate=False,
        path="./results",  # IMPORTANT : doit etre un dossier valide, pas None
    )
    results = sim(sim_obj)
    # results : DataFrame indexe par temps, colonnes BG, CGM, CHO, insulin, LBGI, HBGI, Risk
    return results


# ---------------------------------------------------------------------------
# MODE B : interface Gymnasium (RL-ready) -- pour plus tard
# ---------------------------------------------------------------------------
# Decommente quand tu passeras au RL. step() renvoie (obs, reward, terminated,
# truncated, info). Reward par defaut = risk[t-1] - risk[t].

def make_gym_env(patient_name="adult#001"):
    import gymnasium as gym
    from gymnasium.envs.registration import register

    env_id = f"simglucose/{patient_name.replace('#','')}-v0"
    register(
        id=env_id,
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=480,  # ~1 jour a 3 min/pas
        kwargs={"patient_name": patient_name},
    )
    return gym.make(env_id)


if __name__ == "__main__":
    res = run_direct()
    print(res[["BG", "CGM", "CHO", "insulin", "Risk"]].describe().round(2).to_string())
    print(f"\n{len(res)} pas de temps | BG min/max : "
          f"{res.BG.min():.1f} / {res.BG.max():.1f} mg/dL")
