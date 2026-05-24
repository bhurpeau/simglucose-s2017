# Tâche : porter simglucose du modèle UVA/Padova S2013 vers S2017

## Contexte
`simglucose` (https://github.com/jxx123/simglucose) implémente le modèle
UVA/Padova T1D version S2008/2013. Je veux faire évoluer le cœur dynamique
vers la version S2017, décrite dans :

  Visentin et al. (2018), "The UVA/Padova Type 1 Diabetes Simulator Goes
  From Single Meal to Single Day", J Diabetes Sci Technol 12(2):273-281.
  Équations A1–A28 en appendice. PDF : `docs/visentin2018.pdf`.

Objectif : un ENVIRONNEMENT DE SIMULATION fidèle à S2017. Pas de RL pour
l'instant — il faut juste que la dynamique soit correcte et que l'API
existante (T1DSimEnv, T1DPatient, scénarios, BBController) continue de
fonctionner.

## État actuel du code (déjà repéré)
- Le système d'ODE est dans `simglucose/patient/t1dpatient.py`, méthode
  statique `model(t, x, action, params, last_Qsto, last_foodtaken)`,
  vecteur `dxdt` de 13 états.
- EGP est calculé ~ligne 151 : `EGPt = params.kp1 - params.kp2*x[3] - params.kp3*x[8]`
- Utilisation insulino-dépendante ~ligne 166 : `Vmt = params.Vm0 + params.Vmx*x[6]`
- Les paramètres sont des constantes par sujet, chargées depuis
  `simglucose/params/vpatient_params.csv` (30 sujets : 10 adult, 10
  adolescent, 10 child). Voir aussi `definitions_of_vpatient_parameters.md`.

## Différences S2013 -> S2017 à implémenter
Travaille par étapes, une par commit, avec un test de non-régression
après chaque étape (cf. section Validation).

### Étape 1 — Variabilité intra-journalière de la sensibilité à l'insuline
S2017 rend kp3 (action insuline sur production hépatique) et Vmx (action
insuline sur utilisation tissulaire) TIME-VARYING au cours de la journée
(éq. A5 et A10 du papier ; cf. Visentin et al. 2015 réf.28 pour la
paramétrisation : 7 classes de variabilité circadienne de SI).
- Introduis une notion d'heure-de-la-journée dans `model()` (l'argument `t`
  est en minutes depuis le début ; il faut le relier à l'heure réelle via
  le start_time du scénario).
- Modélise kp3(t) et Vmx(t) comme des multiplicateurs d'un profil journalier.
- Comme les 7 classes et leurs probabilités ne sont PAS dans le PDF,
  paramétrise-les de façon explicite et configurable (un dict ou un petit
  CSV `simglucose/params/si_variability_params.csv`), avec des valeurs
  PLACEHOLDER clairement marquées `# TODO: calibrer sur Visentin 2015`.
  Ne hardcode pas en dur.

### Étape 2 — Phénomène de l'aube (dawn)
Entre 3h00 et 7h00 : augmentation linéaire de kp1 (EGP à glucose/insuline
nuls) et diminution de kir (utilisation insulino-dépendante). Réf. Mallad
et al. (réf.29) : ~+1.5 mg/kg/min d'EGP, ~+30 mg/dL de glucose sur l'intervalle.
- Même approche : profil configurable, valeurs moyennes du papier en
  placeholder documenté.

### Étape 3 — Modèle de glucagon mis à jour (éq. A23–A28)
Deux changements vs S2013 :
1. Clairance du glucagon désormais FRACTIONNAIRE (par unité de volume de
   distribution, mL/kg/min) au lieu d'un taux absolu (min^-1).
2. Pas de sécrétion statique quand glucose >= glucose basal (éq. A25),
   pour éviter les oscillations non physiologiques.
- Vérifie d'abord si le modèle 13 états de ce repo inclut DÉJÀ un
  sous-système glucagon (a priori NON — les 13 états sont glucose/insuline/
  repas, sans alpha-cellules ni glucagon). Si absent, c'est un AJOUT d'états
  (éq. A23–A28 introduisent H, SR_H, H_sc1, H_sc2…), pas une simple modif :
  SIGNALE-LE MOI AVANT d'implémenter, qu'on décide ensemble du périmètre.

## Contraintes
- Rétrocompatibilité : tout sujet sans les nouveaux paramètres doit retomber
  sur le comportement S2013 (modifs time-varying désactivables via un flag
  `model_version="S2013"|"S2017"`, défaut "S2013").
- Ne touche pas à l'API publique (T1DPatient.withName, T1DSimEnv, etc.).
- Code lisible, commenté avec la référence d'équation (ex. `# éq. A5`).
- Toute valeur non issue du PDF = placeholder marqué TODO avec la réf. papier.

## Validation (à exécuter après CHAQUE étape)
1. Test de non-régression : en mode S2013 (défaut), la simulation de
   référence doit reproduire EXACTEMENT `tests/baseline_s2013.csv`
   (généré avant toute modif — voir ce fichier et `tests/make_baseline.py`).
2. Test de plausibilité S2017 : même scénario en mode S2017, vérifie que
   - BG reste physiologique (~70–300 mg/dL avec contrôle),
   - remontée nocturne de BG entre 3h et 7h (effet dawn),
   - réponse à un repas identique différente entre petit-déj et dîner
     (effet variabilité SI).
3. Pas de NaN/inf, pas d'états négatifs persistants.

## Livrables
- Commits séparés par étape, messages clairs.
- Un CHANGELOG.md décrivant ce qui a changé et où sont les placeholders
  à calibrer.
- Un script `examples/run_s2017.py` montrant une simulation en mode S2017.

## Démarrage
Commence par lire `simglucose/patient/t1dpatient.py` en entier et
l'appendice de `docs/visentin2018.pdf`, puis propose-moi ton plan
d'implémentation et confirme le périmètre de l'Étape 3 (glucagon) AVANT
d'écrire du code.
