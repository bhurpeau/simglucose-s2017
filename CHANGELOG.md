# Changelog — portage S2013 → S2017

## [S2017] — 2026-05-24

Portage du modèle UVA/Padova T1D de la version S2013 vers S2017, d'après :
> Visentin et al. (2018), "The UVA/Padova Type 1 Diabetes Simulator Goes From Single
> Meal to Single Day", J Diabetes Sci Technol 12(2):273-281.

### Rétrocompatibilité

Le flag `model_version="S2013"` (défaut) est **entièrement rétrocompatible**.
Non-régression vérifiable par : `python tests/make_baseline.py --check`

---

### Étape 1 — Variabilité intra-journalière de la sensibilité à l'insuline

**Fichiers :** `simglucose/patient/t1dpatient.py`, `simglucose/params/si_variability_params.csv`

- `kp3(t)` et `Vmx(t)` deviennent time-varying (éq. A5 et A10 de Visentin 2018).
- 7 classes SI avec paliers à **04:00, 11:00 et 17:00** (Visentin 2015, Fig. 3).
- Multiplicateur "bas" = **α = 0.4**, bruit journalier **σ = 0.2** (Visentin 2015, données réelles).

| Classe | Pattern | Proba | si_B | si_L | si_D |
|--------|---------|-------|------|------|------|
| 1 | h-h-h | 0.10 | 1.0 | 1.0 | 1.0 |
| 2 | h-h-l | 0.05 | 1.0 | 1.0 | 0.4 |
| 3 | h-l-h | 0.05 | 1.0 | 0.4 | 1.0 |
| 4 | h-l-l | 0.10 | 1.0 | 0.4 | 0.4 |
| 5 | l-h-h | 0.20 | 0.4 | 1.0 | 1.0 |
| 6 | l-h-l | 0.20 | 0.4 | 1.0 | 0.4 |
| 7 | l-l-h | 0.30 | 0.4 | 0.4 | 1.0 |

---

### Étape 2 — Phénomène de l'aube

**Fichiers :** `simglucose/patient/t1dpatient.py`, `simglucose/params/dawn_params.csv`

- `kp1(t)` : rampe linéaire +**1.5 mg/kg/min** sur 03:00–07:00 (éq. A5). Source : Mallad et al. (réf. 29).
- `Vmx(t)` réduit par `delta_kir_max` sur la même fenêtre (éq. A10).
  **⚠ TODO : calibrer `delta_kir_max=0.3` sur Mallad et al. (réf. 29) — placeholder.**

---

### Étape 3 — Glucagon mis à jour (éq. A23-A28)

**Fichiers :** `simglucose/patient/t1dpatient.py`

Le modèle S2013 de ce repo ne contenait pas de glucagon. S2017 ajoute **3 états ODE**
(vecteur 13 → 16 états en mode S2017) :

| Index | État | Éq. | Description |
|-------|------|-----|-------------|
| 13 | `X_H` | A8 | Action glucagon sur EGP |
| 14 | `H` | A23 | Glucagon plasmatique |
| 15 | `SR_Hs` | A25 | Sécrétion statique glucagon |

`SR_Hd` (éq. A26) est algébrique : `δ · max(−dG/dt, 0)`.

EGP S2017 : `EGP(t) = kp1(t) - kp2·G - kp3(t)·X_L + ξ·X_H` (éq. A5)

Corrections vs S2013 :
1. Clairance glucagon **fractionnaire** `n` (mL/kg/min) — éq. A23.
2. **Pas de sécrétion statique** quand G ≥ G_b — éq. A25.

**⚠ Tous les paramètres glucagon sont des placeholders (TODO: calibrer) :**
`n=0.142`, `rho=0.00457`, `sigma_max=0.0082`, `SR_Hb=0.003`, `H_b=0.065`,
`k_H=0.0465`, `xi=0.0065`, `delta=0.0045`.

---

### Validation

```
python tests/make_baseline.py --check                   # non-régression S2013
python -m pytest tests/test_s2017_plausibility.py -v    # 7/7 tests verts
python examples/run_s2017.py --compare                  # démo S2017 vs S2013
```

---

Baseline de non-régression S2013 : `tests/baseline_s2013.csv`
