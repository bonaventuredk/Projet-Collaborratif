"""
OptiBobineOptuna.py
===================
Optimisation des paramètres de la bobine et du tube par méthode Bayésienne
(Optuna — TPE : Tree-structured Parzen Estimator).

Pourquoi Optuna plutôt que Scipy DE ?
--------------------------------------
L'Évolution Différentielle est une méthode populationnelle : elle évalue une
génération entière (pop_size × n_dim ≈ 84 candidats) avant d'apprendre quoi
que ce soit sur le paysage de la fonction objectif.
Optuna–TPE construit un modèle probabiliste dès le premier essai et concentre
les nouveaux essais dans les zones prometteuses. Pour des simulations
coûteuses (comme ici), le gain de temps jusqu'à une bonne solution est typiquement
5 à 10 fois plus élevé.

Avantages supplémentaires d'Optuna :
  ✓ Reprise après interruption (stockage SQLite)
  ✓ Pruning (arrêt prématuré des essais mauvais)
  ✓ Parallélisme nativement supporté (plusieurs workers SQLite)
  ✓ Visualisation intégrée (courbes de convergence, importances des params)
  ✓ Gestion native des entiers, flottants et catégories
  ✓ Pas d'installation supplémentaire lourde (pip install optuna)

Positions initiales fixes
--------------------------
Quand `--fixed` est activé (option par défaut), les particules sont placées
sur une grille régulière y ∈ [marge, D−marge] au lieu de positions aléatoires.
La simulation devient alors DÉTERMINISTE : le même vecteur de paramètres donne
toujours le même score, ce qui élimine le bruit stochastique et supprime le
besoin de moyenner plusieurs runs (N_RUNS = 1 suffit).

Paramètres optimisés (7 dimensions)
-------------------------------------
  Nom             │ Type        │ Borne basse  │ Borne haute
  ────────────────┼─────────────┼──────────────┼─────────────────
  L               │ catégoriel  │ (liste des L disponibles)
  I               │ continu     │ 0.1 A        │ 10 A
  R_coil          │ continu     │ D / 4        │ 2 × D
  N_spires        │ entier      │ 10           │ 500
  spacing         │ continu log │ 0.5 mm       │ 10 mm
  x_coil_frac     │ continu     │ 0.0          │ 0.8  (fraction de L)
  B0              │ continu log │ 1e-8 T       │ 1e-4 T

Usage
-----
    pip install optuna
    python OptiBobineOptuna.py                         # 200 essais, positions fixes
    python OptiBobineOptuna.py --trials 500            # plus d'essais
    python OptiBobineOptuna.py --no-fixed              # positions aléatoires (+ bruit)
    python OptiBobineOptuna.py --particles 30          # moins de particules (+ rapide)
    python OptiBobineOptuna.py --study my_study        # nom d'étude personnalisé
    python OptiBobineOptuna.py --resume                # reprend une étude existante
    python OptiBobineOptuna.py --plot                  # affiche les graphiques Optuna
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# ── Vérification de la disponibilité d'Optuna ────────────────────────────────
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # on gère l'affichage nous-mêmes
except ImportError:
    print("\n[ERREUR] Optuna n'est pas installé.")
    print("  Installez-le avec :  pip install optuna")
    print("  Puis relancez ce script.")
    sys.exit(1)

from SolveParticule6 import (
    run_multi_lap, make_fixed_y_positions,
    D, lwall, eta, Rtip, Lwall,
    load_freefem_data,
)
from scipy.interpolate import CloughTocher2DInterpolator

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DÉTECTION DES LONGUEURS ET PRÉCHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def discover_available_lengths(folder="."):
    """Trouve les L pour lesquels les trois fichiers FreeFem existent."""
    import re
    lengths = []
    for fname in os.listdir(folder):
        m = re.match(r"ux([\d.]+)\.txt$", fname)
        if m:
            L = float(m.group(1))
            has_uy    = os.path.exists(os.path.join(folder, f"uy{L}.txt"))
            has_nodes = os.path.exists(os.path.join(folder, f"nodes{L}.txt"))
            if has_uy and has_nodes:
                lengths.append(L)
            else:
                missing = []
                if not has_uy:    missing.append(f"uy{L}.txt")
                if not has_nodes: missing.append(f"nodes{L}.txt")
                print(f"  [WARN] L={L:.4f} m ignoré — fichier(s) manquant(s) : "
                      f"{', '.join(missing)}")
    return sorted(lengths)


def preload_all_interpolators(available_lengths, folder="."):
    """Construit le cache  { L : (ux_interp, uy_interp) }."""
    cache = {}
    for L in available_lengths:
        try:
            nodes  = np.loadtxt(os.path.join(folder, f"nodes{L}.txt"))
            ux_dat = np.loadtxt(os.path.join(folder, f"ux{L}.txt"))
            uy_dat = np.loadtxt(os.path.join(folder, f"uy{L}.txt"))
            cache[L] = (
                CloughTocher2DInterpolator(nodes, ux_dat),
                CloughTocher2DInterpolator(nodes, uy_dat),
            )
            print(f"  ✓  L = {L:.4f} m  ({len(nodes)} nœuds)")
        except Exception as exc:
            print(f"  ✗  L = {L:.4f} m  ERREUR : {exc}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FONCTION OBJECTIF OPTUNA
# ─────────────────────────────────────────────────────────────────────────────

class DesalinisationTrial:
    """
    Functor compatible avec optuna.study.optimize().

    Chaque appel `__call__(trial)` construit un dict `bobine` à partir des
    suggestions d'Optuna, lance la simulation et retourne l'efficacité à
    MAXIMISER.
    """

    def __init__(self, interp_cache, available_L,
                 n_na, n_cl, n_h2o,
                 n_laps, n_runs, dt,
                 use_fixed_positions):
        self.interp_cache          = interp_cache
        self.available_L           = available_L
        self.n_na                  = n_na
        self.n_cl                  = n_cl
        self.n_h2o                 = n_h2o
        self.n_laps                = n_laps
        self.n_runs                = n_runs
        self.dt                    = dt
        self.use_fixed_positions   = use_fixed_positions

        # Positions fixes précalculées une fois pour toutes
        if use_fixed_positions:
            fixed_y = make_fixed_y_positions(max(n_na, n_cl))
            self.fixed_positions = {
                'Na+': fixed_y,
                'Cl-': fixed_y,
                'H20': np.array([]),
            }
        else:
            self.fixed_positions = None

        # Compteurs d'affichage
        self._trial_count = 0
        self._best_score  = 0.0

    def __call__(self, trial: "optuna.Trial") -> float:
        """Propose les paramètres, lance la simulation, retourne le score."""
        self._trial_count += 1

        # ── Suggestion des paramètres par TPE ────────────────────────────────

        # L est traité comme une catégorie : TPE choisit parmi les valeurs
        # disponibles sans supposer d'ordre numérique.
        L = trial.suggest_categorical("L", self.available_L)

        I        = trial.suggest_float("I",        0.1,  10.0)
        R_coil   = trial.suggest_float("R_coil",   D/4,  3*D/4)
        N_spires = trial.suggest_int  ("N_spires",  10,   250)

        # Paramètre log-uniforme : l'espacement et B0 varient sur plusieurs
        # ordres de grandeur → meilleure exploration en espace log
        spacing  = trial.suggest_float("spacing",  5e-4,  1e-2, log=True)
        B0       = trial.suggest_float("B0",       1e-8,  1e-4, log=True)

        # x_coil en fraction de L (évite de devoir changer les bornes selon L)
        x_coil_frac = trial.suggest_float("x_coil_frac", 0.0, 0.8)
        x_coil      = x_coil_frac * L

        y_coil= trial.suggest_float("x_coil_frac", 0.0, D)

        z_coil = trial.suggest_float("x_coil_frac", -1.0, 1.0)

        bobine = {
            "L"        : L,
            "I"        : I,
            "Rayon"    : R_coil,
            "Nb_spire" : N_spires,
            "spacing"  : spacing,
            "x_coil"   : x_coil,
            "y_coil"   : y_coil,
            "z_coil"   : z_coil,
            "B0"       : B0,
        }

        ux_interp, uy_interp = self.interp_cache.get(L, (None, None))
        n_particles = {'Na+': self.n_na, 'Cl-': self.n_cl, 'H20': self.n_h2o}
        total_ions  = self.n_na + self.n_cl   # H2O ne compte pas dans l'efficacité

        # ── Simulation ───────────────────────────────────────────────────────
        # Avec positions fixes : déterministe → 1 run suffit (n_runs ignoré)
        # Avec positions aléatoires : on moyenne sur n_runs pour réduire le bruit
        actual_runs = 1 if self.use_fixed_positions else self.n_runs

        total_Na = 0.0
        total_Cl = 0.0

        for run_idx in range(actual_runs):
            try:
                _, _, bilan = run_multi_lap(
                    n_particles, self.n_laps, self.dt,
                    bobine,
                    verbose         = False,
                    preloaded_interps = (ux_interp, uy_interp),
                    fixed_positions   = self.fixed_positions,
                )
                for lap_bilan in bilan:
                    total_Na += lap_bilan["bas (de Na+)"]["Na+"]
                    total_Cl += lap_bilan["haut (de Cl-)"]["Cl-"]

            except Exception as exc:
                # Si la simulation plante (paramètres extrêmes), retourner 0
                trial.set_user_attr("error", str(exc))
                return 0.0

        denom = actual_runs * self.n_laps * total_ions
        score = (total_Na + total_Cl) / denom if denom > 0 else 0.0

        # ── Affichage ─────────────────────────────────────────────────────────
        if score > self._best_score:
            self._best_score = score
            mode = "FIXE" if self.use_fixed_positions else "ALÉA"
            print(
                f"[#{self._trial_count:4d}/{mode}] ★ NOUVEAU MEILLEUR  "
                f"score={score:.4f} ({score*100:.1f}%)"
                f"  L={L:.4f}m  I={I:.2f}A  R={R_coil*1e3:.1f}mm"
                f"  N={N_spires}  sp={spacing*1e3:.2f}mm"
                f"  xc={x_coil*1e3:.1f}mm  yc={y_coil*1e3:.1f}mm"
                f"  zc={z_coil*1e3:.1f}mm  B0={B0:.2e}T"
            )
        elif self._trial_count % 25 == 0:
            print(f"[#{self._trial_count:4d}]   score={score:.4f}"
                  f"  (meilleur={self._best_score:.4f})")

        return score


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AFFICHAGE DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────

def print_results(study: "optuna.Study"):
    """Affiche un résumé des meilleurs paramètres trouvés."""
    best = study.best_trial
    print()
    print("=" * 70)
    print("  RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"\nMeilleure efficacité de séparation : {best.value:.4f}  "
          f"({best.value * 100:.1f} %)")
    print(f"Essai n°                           : {best.number}")
    print(f"Nombre total d'essais              : {len(study.trials)}")

    p = best.params
    L = p["L"]
    print("\nParamètres optimaux :")
    print(f"  Longueur du tube  L       = {L:.4f} m  ({L*100:.2f} cm)")
    print(f"  Courant           I       = {p['I']:.3f} A")
    print(f"  Rayon bobine      R_coil  = {p['R_coil']*1e3:.2f} mm")
    print(f"  Nombre de spires  N       = {p['N_spires']}")
    print(f"  Espacement        spacing = {p['spacing']*1e3:.3f} mm")
    print(f"  Position x bobine x_coil = {p['x_coil_frac']*L*1e3:.2f} mm"
          f"  ({p['x_coil_frac']*100:.1f} % de L)")
    print(f"  Normalisation     B0      = {p['B0']:.3e} T")

    # Top 5 pour comparaison
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )
    if len(trials_sorted) > 1:
        print(f"\nTop 5 des essais :")
        print(f"  {'Rang':>4}  {'Score':>8}  {'L':>8}  {'I':>7}  {'N':>5}  {'R_coil':>9}")
        print("  " + "─" * 55)
        for rank, t in enumerate(trials_sorted[:5], 1):
            tp = t.params
            print(f"  {rank:4d}  {t.value:8.4f}  "
                  f"{tp['L']:8.4f}  {tp['I']:7.3f}  "
                  f"{tp['N_spires']:5d}  {tp['R_coil']*1e3:9.2f}mm")

    # Importance des paramètres (nécessite plusieurs essais)
    if len(study.trials) >= 20:
        try:
            importances = optuna.importance.get_param_importances(study)
            print(f"\nImportance relative des paramètres :")
            for name, imp in importances.items():
                bar = "█" * int(imp * 30)
                print(f"  {name:15s} {bar:<30s} {imp:.3f}")
        except Exception:
            pass  # peut échouer si pas assez de diversité


def save_results(study: "optuna.Study", args):
    """Sauvegarde les résultats dans un fichier JSON."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"opti_results_optuna_{ts}.json"

    best = study.best_trial
    history = []
    for t in study.trials:
        if t.value is not None:
            history.append({
                "trial"  : t.number,
                "score"  : round(t.value, 6),
                "params" : {k: (float(v) if isinstance(v, float) else v)
                            for k, v in t.params.items()},
            })

    save_data = {
        "best_efficiency_fraction": best.value,
        "best_efficiency_percent" : round(best.value * 100, 2),
        "best_trial_number"       : best.number,
        "best_params"             : {k: (float(v) if isinstance(v, float) else v)
                                     for k, v in best.params.items()},
        "optimizer_config"        : vars(args),
        "total_trials"            : len(study.trials),
        "history"                 : history,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés dans : {out_file}")
    return out_file


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimisation bobine désalinisateur — Optuna TPE (Bayésien)"
    )
    parser.add_argument("--trials",    type=int,   default=200,
                        help="Nombre d'essais Optuna (défaut: 200)")
    parser.add_argument("--runs",      type=int,   default=3,
                        help="Runs stochastiques par essai si --no-fixed (défaut: 3)")
    parser.add_argument("--particles", type=int,   default=50,
                        help="Particules par espèce (défaut: 50)")
    parser.add_argument("--laps",      type=int,   default=1,
                        help="Tours simulés par essai (défaut: 1)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Graine aléatoire (défaut: 42)")
    parser.add_argument("--folder",    type=str,   default=".",
                        help="Dossier contenant les fichiers FreeFem (défaut: .)")
    parser.add_argument("--study",     type=str,   default="desalinisation",
                        help="Nom de l'étude Optuna (défaut: desalinisation)")
    parser.add_argument("--no-fixed",  dest="fixed", action="store_false",
                        help="Désactive les positions fixes (ajoute du bruit)")
    parser.add_argument("--resume",    action="store_true",
                        help="Reprend une étude SQLite existante (même --study)")
    parser.add_argument("--plot",      action="store_true",
                        help="Affiche les graphiques Optuna après l'optimisation")
    parser.set_defaults(fixed=True)
    args = parser.parse_args()

    # ── Bannière ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  OPTIMISATION BOBINE DÉSALINISATEUR")
    print("  Méthode : Optuna — TPE (Tree-structured Parzen Estimator)")
    mode_str = "positions initiales FIXES (déterministe)" if args.fixed \
               else f"positions aléatoires, {args.runs} runs/essai (stochastique)"
    print(f"  Mode   : {mode_str}")
    print("=" * 70)

    # ── Détection des longueurs disponibles ──────────────────────────────────
    print("\nDétection des fichiers FreeFem disponibles...")

    os.system('FreeFem++ MaillageStokesAda.edp')
    available_L = discover_available_lengths(args.folder)

    if not available_L:
        print("\n[ERREUR] Aucun triplet (ux/uy/nodes){L}.txt trouvé dans :", args.folder)
        sys.exit(1)

    print(f"\n{len(available_L)} longueur(s) disponible(s) :")
    for L in available_L:
        print(f"  L = {L:.4f} m  ({L*100:.2f} cm)")

    # ── Préchargement des interpolateurs ─────────────────────────────────────
    print("\nChargement des interpolateurs FreeFem (opération unique)...")
    interp_cache = preload_all_interpolators(available_L, args.folder)
    if not interp_cache:
        print("\n[ERREUR] Aucun interpolateur chargé.")
        sys.exit(1)
    available_L = sorted(interp_cache.keys())

    # ── Configuration ─────────────────────────────────────────────────────────
    #available_L = 0.057
    N_NA  = args.particles
    N_CL  = args.particles
    N_H2O = 0
    DT    = 1e-3

    print(f"\nConfiguration :")
    print(f"  Particules par espèce        : {args.particles}")
    print(f"  Tours simulés par essai      : {args.laps}")
    if args.fixed:
        print(f"  Positions initiales          : FIXES (grille régulière, déterministe)")
        print(f"  Runs par essai               : 1 (déterministe → 1 suffit)")
    else:
        print(f"  Positions initiales          : ALÉATOIRES")
        print(f"  Runs par essai (moyennage)   : {args.runs}")
    print(f"  Essais Optuna                : {args.trials}")
    print(f"  Nom de l'étude               : {args.study}")

    # ── Création / reprise de l'étude Optuna ─────────────────────────────────
    db_file  = f"{args.study}.db"
    storage  = f"sqlite:///{db_file}"

    if args.resume and os.path.exists(db_file):
        print(f"\nReprise de l'étude existante : {db_file}")
        study = optuna.load_study(
            study_name = args.study,
            storage    = storage,
            sampler    = TPESampler(seed=args.seed),
        )
        existing_trials = len([t for t in study.trials if t.value is not None])
        print(f"  {existing_trials} essais déjà effectués.")
    else:
        study = optuna.create_study(
            study_name    = args.study,
            direction     = "maximize",   # on veut MAXIMISER l'efficacité
            storage       = storage,
            load_if_exists= args.resume,
            sampler       = TPESampler(
                seed                    = args.seed,
                n_startup_trials        = 20,   # exploration aléatoire initiale
                multivariate            = True, # tient compte des corrélations
                consider_prior          = True,
                consider_magic_clip     = True,
            ),
        )

    # ── Construction du functor objectif ─────────────────────────────────────
    objective = DesalinisationTrial(
        interp_cache        = interp_cache,
        available_L         = available_L,
        n_na                = N_NA,
        n_cl                = N_CL,
        n_h2o               = N_H2O,
        n_laps              = args.laps,
        n_runs              = args.runs,
        dt                  = DT,
        use_fixed_positions = args.fixed,
    )

    # ── Lancement de l'optimisation ───────────────────────────────────────────
    print(f"\nDémarrage de l'optimisation ({args.trials} essais)...\n")

    study.optimize(
        objective,
        n_trials         = args.trials,
        show_progress_bar= False,  # on affiche nous-mêmes les nouveaux meilleurs
        gc_after_trial   = True,   # libère la mémoire entre les essais
    )

    # ── Affichage et sauvegarde ───────────────────────────────────────────────
    print_results(study)
    out_file = save_results(study, args)

    print(f"\nL'étude est persistée dans : {db_file}")
    print(f"  → Pour reprendre plus tard : python OptiBobineOptuna.py "
          f"--resume --study {args.study} --trials {args.trials}")

    # ── Visualisations Optuna ─────────────────────────────────────────────────
    if args.plot:
        try:
            import optuna.visualization as vis
            import matplotlib.pyplot as plt

            # Courbe de convergence
            fig1 = vis.plot_optimization_history(study)
            fig1.show()

            # Importance des hyperparamètres
            if len(study.trials) >= 20:
                fig2 = vis.plot_param_importances(study)
                fig2.show()

            # Relations entre paramètres et score
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.show()

            print("\nGraphiques Optuna affichés (fermer les fenêtres pour quitter).")
            plt.show()
        except Exception as exc:
            print(f"\n[WARN] Impossible d'afficher les graphiques Optuna : {exc}")
            print("  Installez plotly :  pip install plotly")

    return study


# ─────────────────────────────────────────────────────────────────────────────
# 5.  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    study = main()
