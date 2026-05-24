from .base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources

logger = logging.getLogger(__name__)

Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["Gsub"])

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)
SI_VARIABILITY_FILE = pkg_resources.resource_filename(
    "simglucose", "params/si_variability_params.csv"
)
DAWN_PARAMS_FILE = pkg_resources.resource_filename(
    "simglucose", "params/dawn_params.csv"
)

# SI transition times from Visentin 2015 (min from midnight).
# The SI profile steps at 04:00 (t_B), 11:00 (t_L), and 17:00 (t_D).
_SI_T_B = 240   # 04:00
_SI_T_L = 660   # 11:00
_SI_T_D = 1020  # 17:00

# S2017 glucagon state indices (states 13-15, appended after the original 13).
_IDX_X_H  = 13  # glucagon action on EGP (éq. A8)
_IDX_H    = 14  # plasma glucagon concentration (éq. A23)
_IDX_SR_HS = 15  # static glucagon secretion (éq. A25)


def _load_si_variability():
    """Return SI variability params as a dict keyed by class_id (int)."""
    df = pd.read_csv(SI_VARIABILITY_FILE, comment="#")
    result = {}
    for _, row in df.iterrows():
        result[int(row["class_id"])] = {
            "prob": row["prob"],
            "si_B": row["si_B"],
            "si_L": row["si_L"],
            "si_D": row["si_D"],
            "sigma": row["sigma_noise"],
        }
    return result


def _load_dawn_params():
    """Return dawn params as a dict {param: value}."""
    df = pd.read_csv(DAWN_PARAMS_FILE, comment="#")
    return dict(zip(df["param"], df["value"]))


class T1DPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(
        self,
        params,
        init_state=None,
        random_init_bg=False,
        seed=None,
        t0=0,
        model_version="S2013",
        start_time=360,
        si_class=None,
    ):
        """
        T1DPatient constructor.

        Parameters
        ----------
        params : pandas.Series
            Subject-specific parameters.
        init_state : array-like, optional
            Initial state vector. Default: from params.
        random_init_bg : bool
            Randomise glucose-related initial states.
        seed : int or None
            RNG seed (for random_init_bg and SI noise).
        t0 : float
            Simulation start time (min). Default 0.
        model_version : {"S2013", "S2017"}
            "S2013" (default) gives identical dynamics to the original code.
            "S2017" enables time-varying SI, dawn phenomenon, and glucagon.
        start_time : int
            Real time of day when t=0, in minutes from midnight (default 360
            = 06:00). Used by S2017 to compute circadian phase.
        si_class : int or None
            SI variability class 1-7 (Visentin 2015). If None and model_version
            is "S2017", drawn randomly from the class probability distribution.
        """
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        self.model_version = model_version
        self.start_time = start_time  # min from midnight at t=0
        self._si_class_override = si_class
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id.
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # ODE solver
        self._odesolver.set_f_params(
            action,
            self._params,
            self._last_Qsto,
            self._last_foodtaken,
            self._s2017_cfg,
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    # ------------------------------------------------------------------
    # S2017 helper: SI profile
    # ------------------------------------------------------------------
    @staticmethod
    def _si_multiplier(t_of_day, si_profile):
        """
        Step-wise SI multiplier for kp3 and Vmx.
        Transitions at 04:00, 11:00, 17:00 (Visentin 2015).
        Noise already baked into si_profile values.
        """
        if t_of_day < _SI_T_B or t_of_day >= _SI_T_D:
            # pre-breakfast or late evening: dinner-level SI
            return si_profile["si_D_actual"]
        elif t_of_day < _SI_T_L:
            # breakfast period
            return si_profile["si_B_actual"]
        else:
            # lunch period
            return si_profile["si_L_actual"]

    @staticmethod
    def _dawn_scale(t_of_day, dawn_cfg):
        """
        Linear ramp from 0→1 between dawn_start and dawn_end (éq. A5).
        Returns alpha in [0, 1].
        """
        start = dawn_cfg["dawn_start"]
        end = dawn_cfg["dawn_end"]
        if t_of_day < start or t_of_day >= end:
            return 0.0
        return (t_of_day - start) / (end - start)

    # ------------------------------------------------------------------
    # Main ODE model
    # ------------------------------------------------------------------
    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken, s2017_cfg=None):
        """
        ODE right-hand side.

        In S2013 mode (s2017_cfg is None or model_version=="S2013"), this is
        identical to the original simglucose implementation.

        In S2017 mode the following extensions are active:
          - kp3(t) and Vmx(t) are time-varying (éq. A5, A10, Visentin 2015)
          - kp1(t) increases and Vmx(t) decreases during 3-7 am (éq. A5, A10)
          - Glucagon subsystem: X_H, H, SR_Hs (éq. A8, A23-A26)
            EGP gains the term +ξ·X_H (éq. A5)
        """
        mv = "S2013" if (s2017_cfg is None) else s2017_cfg.get("model_version", "S2013")
        n_states = 16 if mv == "S2017" else 13
        dxdt = np.zeros(n_states)

        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # ----------------------------------------------------------------
        # S2017 time-varying parameters
        # ----------------------------------------------------------------
        if mv == "S2017":
            t_of_day = (t + s2017_cfg["start_time"]) % 1440  # min from midnight
            si_mult = T1DPatient._si_multiplier(t_of_day, s2017_cfg["si_profile"])  # éq. A5, A10
            alpha_dawn = T1DPatient._dawn_scale(t_of_day, s2017_cfg["dawn_cfg"])

            dawn = s2017_cfg["dawn_cfg"]
            kp1_t  = params.kp1 + alpha_dawn * dawn["delta_kp1_max"]   # éq. A5
            kp3_t  = params.kp3 * si_mult                               # éq. A5
            Vmx_t  = params.Vmx * si_mult * (1.0 - alpha_dawn * dawn["delta_kir_max"])  # éq. A10
        else:
            kp1_t = params.kp1
            kp3_t = params.kp3
            Vmx_t = params.Vmx

        # ----------------------------------------------------------------
        # Glucose in the stomach
        # ----------------------------------------------------------------
        qsto = x[0] + x[1]
        # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
        # from mg to g. See https://github.com/jxx123/simglucose/issues/41
        Dbar = last_Qsto + last_foodtaken * 1000  # mg

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW

        # ----------------------------------------------------------------
        # Endogenous Glucose Production (éq. A5)
        # kp1(t) time-varying in S2017 (dawn); kp3(t) time-varying (SI)
        # ----------------------------------------------------------------
        EGPt = kp1_t - params.kp2 * x[3] - kp3_t * x[8]
        if mv == "S2017":
            # Glucagon action on EGP: +ξ·X_H (éq. A5, A8)
            xi = s2017_cfg["glucagon_cfg"]["xi"]
            EGPt = EGPt + xi * x[_IDX_X_H]

        # Glucose Utilization (insulin-independent)
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        # ----------------------------------------------------------------
        # Glucose utilization — insulin-dependent (éq. A10)
        # Vmx(t) time-varying in S2017 (SI variability + dawn kir)
        # ----------------------------------------------------------------
        Vmt = params.Vm0 + Vmx_t * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = (
            -(params.m2 + params.m4) * x[5]
            + params.m1 * x[9]
            + params.ka1 * x[10]
            + params.ka2 * x[11]
        )
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization (éq. A11)
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production — double delay (éq. A6-A7)
        dxdt[7] = -params.ki * (x[7] - It)
        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        # ----------------------------------------------------------------
        # S2017 glucagon subsystem (éq. A8, A23-A26)
        # States: x[13]=X_H, x[14]=H, x[15]=SR_Hs
        # ----------------------------------------------------------------
        if mv == "S2017":
            gcfg = s2017_cfg["glucagon_cfg"]
            H_b    = gcfg["H_b"]
            n_H    = gcfg["n"]        # fractional clearance (mL/kg/min) — éq. A23
            rho    = gcfg["rho"]      # rate constant for static secretion — éq. A25
            sigma_max = gcfg["sigma_max"]  # max static secretion — éq. A25
            SR_Hb  = gcfg["SR_Hb"]   # basal glucagon secretion
            k_H    = gcfg["k_H"]     # glucagon action rate constant — éq. A8
            delta  = gcfg["delta"]   # dynamic secretion amplitude — éq. A26

            X_H   = x[_IDX_X_H]    # glucagon action on EGP
            H     = x[_IDX_H]      # plasma glucagon
            SR_Hs = x[_IDX_SR_HS]  # static glucagon secretion

            G_b = params.Gb  # basal plasma glucose (mg/dL)
            G   = x[3] / params.Vg  # plasma glucose (mg/dL)

            # éq. A8: glucagon action on EGP
            dxdt[_IDX_X_H] = -k_H * X_H + k_H * max(H - H_b, 0)

            # éq. A26: dynamic glucagon secretion (algebraic)
            dGp_dt = dxdt[3]
            dG_dt  = dGp_dt / params.Vg  # convert Gp rate to G rate (dL/kg/min → mg/dL/min)
            SR_Hd  = delta * max(-dG_dt, 0)  # secretion when glucose is falling

            # éq. A25: static glucagon secretion — no static secretion when G >= G_b
            if G >= G_b:
                # S2017: no static secretion when glucose is at or above basal
                target_SR_Hs = SR_Hb
            else:
                # Hypoglycemia: static secretion increases
                target_SR_Hs = SR_Hb + sigma_max * (G_b - G) / G_b
            dxdt[_IDX_SR_HS] = -rho * (SR_Hs - target_SR_Hs)

            # éq. A24: total secretion
            SR_H = SR_Hs + SR_Hd

            # éq. A23: plasma glucagon (Ra_H = 0, no exogenous glucagon)
            dxdt[_IDX_H] = -n_H * H + SR_H

        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))

        return dxdt

    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        TODO: add heart rate as an observation
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg
        observation = Observation(Gsub=Gsub)
        return observation

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def _build_s2017_cfg(self, rng):
        """Build the S2017 configuration dict for the ODE solver."""
        if self.model_version == "S2013":
            return None

        # -- SI variability --
        si_table = _load_si_variability()
        if self._si_class_override is not None:
            si_class = int(self._si_class_override)
        else:
            probs = [si_table[k]["prob"] for k in sorted(si_table)]
            si_class = int(rng.choice(sorted(si_table), p=probs))
        cls = si_table[si_class]

        # Apply per-period multiplicative noise (Visentin 2015: N(1, sigma^2))
        sigma = cls["sigma"]
        noise_B = max(0.1, rng.normal(1.0, sigma))
        noise_L = max(0.1, rng.normal(1.0, sigma))
        noise_D = max(0.1, rng.normal(1.0, sigma))

        si_profile = {
            "si_B_actual": np.clip(cls["si_B"] * noise_B, 0.05, 2.0),
            "si_L_actual": np.clip(cls["si_L"] * noise_L, 0.05, 2.0),
            "si_D_actual": np.clip(cls["si_D"] * noise_D, 0.05, 2.0),
            "si_class": si_class,
        }
        logger.info("S2017: SI class %d (%s), multipliers B=%.2f L=%.2f D=%.2f",
                    si_class, cls.get("description", ""),
                    si_profile["si_B_actual"], si_profile["si_L_actual"],
                    si_profile["si_D_actual"])

        # -- Dawn phenomenon --
        dawn_raw = _load_dawn_params()
        dawn_cfg = {k: float(v) for k, v in dawn_raw.items()}

        # -- Glucagon --
        # All parameter values below are placeholders.
        # TODO: calibrate n, rho, sigma_max, SR_Hb, H_b, k_H, xi, delta
        # from Dalla Man et al. 2014 (ref.7) and S2017 supplementary data.
        glucagon_cfg = {
            "n":         0.142,   # TODO: placeholder — fractional clearance (mL/kg/min), éq. A23
            "rho":       0.00457, # TODO: placeholder — static secretion rate (min^-1), éq. A25
            "sigma_max": 0.0082,  # TODO: placeholder — max static secretion (pmol/kg/min), éq. A25
            "SR_Hb":     0.0030,  # TODO: placeholder — basal static secretion (pmol/kg/min), éq. A25
            "H_b":       0.0650,  # TODO: placeholder — basal plasma glucagon (pmol/L), éq. A23
            "k_H":       0.0465,  # TODO: placeholder — glucagon action rate (min^-1), éq. A8
            "xi":        0.0065,  # TODO: placeholder — glucagon effect on EGP (mg/kg/min per pmol/L), éq. A5
            "delta":     0.0045,  # TODO: placeholder — dynamic secretion gain (pmol/kg per mg/dL), éq. A26
        }

        return {
            "model_version": "S2017",
            "start_time": self.start_time,
            "si_profile": si_profile,
            "dawn_cfg": dawn_cfg,
            "glucagon_cfg": glucagon_cfg,
        }

    def reset(self):
        """Reset the patient state to default initial state."""
        rng = np.random.RandomState(self._seed)

        if self._init_state is None:
            init13 = np.copy(self._params.iloc[2:15].values)
        else:
            init13 = np.array(self._init_state, dtype=float)

        if self.random_init_bg:
            mean = [1.0 * init13[3], 1.0 * init13[4], 1.0 * init13[12]]
            cov = np.diag([0.1 * init13[3], 0.1 * init13[4], 0.1 * init13[12]])
            bg_init = rng.multivariate_normal(mean, cov)
            init13[3]  = float(bg_init[0])
            init13[4]  = float(bg_init[1])
            init13[12] = float(bg_init[2])

        # Build S2017 config (uses rng — must come after random_init_bg)
        self._s2017_cfg = self._build_s2017_cfg(rng)

        if self.model_version == "S2017":
            # Append glucagon initial conditions (éq. A8, A23, A25)
            gcfg = self._s2017_cfg["glucagon_cfg"]
            x_H_0   = 0.0           # X_H(0) = 0 (éq. A8)
            H_0     = gcfg["H_b"]   # H(0) = H_b (éq. A23)
            SR_Hs_0 = gcfg["SR_Hb"] # SR_Hs(0) = SR_Hb (éq. A25)
            self.init_state = np.concatenate([init13, [x_H_0, H_0, SR_Hs_0]])
        else:
            self.init_state = init13

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name

        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    p = T1DPatient.withName("adolescent#001")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    t = []
    CHO = []
    insulin = []
    BG = []
    while p.t < 1000:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()
