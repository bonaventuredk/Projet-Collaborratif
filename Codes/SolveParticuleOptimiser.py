"""
Optimisation des paramètres de la bobine par grid search.

À chaque itération :
  - on lance la simulation multi-tours avec un jeu de paramètres
  - on calcule un score de séparation Na+/Cl-
  - on conserve les meilleurs paramètres

Auteurs : bonaventure & audrey & thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import csv
import os
from scipy.special import ellipk, ellipe
from scipy.interpolate import CloughTocher2DInterpolator
from concurrent.futures import ProcessPoolExecutor

# ============================================================
# LISTES DES VALEURS À EXPLORER (modifier selon vos besoins)
# ============================================================

PARAM_GRID = {
    'N':       [30, 40, 50],           # Nombre de spires
    'I':       [0.005, 0.001, 0.01, 0.05],        # Intensité (A)
    'R_coil':  [0.0001, 0.0005, 0.002],   # Rayon des spires (m)
    'spacing': [0.0005,  0.0025 ,0.001],     # Espacement entre spires (m)
    'z_coil':  [0.01, 0.0, 0.005],       # Position axiale centre bobine (m)
    'L':       [0.020, 0.027, 0.035, 0.045],      # Longueur totale du tube (m)
}

max_iter = 972

# ============================================================
# CRITÈRE : score de séparation Na+/Cl-
# score = (Cl- en haut) + (Na+ en bas) - (particules au milieu)
# ============================================================

def compute_score(bilan_detail):
    """
    Calcule le score de séparation à partir du bilan final multi-tours.
    On somme sur tous les tours.
    """
    score = 0
    for stats in bilan_detail:
        score += stats.get("haut (de Cl-)", {}).get("Cl-", 0)
        score += stats.get("bas (de Na+)", {}).get("Na+", 0)
        score -= stats.get("milieu", {}).get("Na+", 0)
        score -= stats.get("milieu", {}).get("Cl-", 0)
        score -= stats.get("milieu", {}).get("H20", 0)
    return score


# ============================================================
# PARAMÈTRES GÉOMÉTRIQUES FIXES
# ============================================================

D       = 0.00276
lwall   = D / 10
Lwall_ratio = 1 / 8    # Lwall = L / 8
eta     = D / 10
Rtip    = eta / 2
delta   = 6 * Rtip
mu0     = 4 * np.pi * 1e-7
B0      = 3e-7

# ============================================================
# CHARGEMENT DES DONNÉES FREEFEM (une seule fois)
# ============================================================

def load_freefem_data(nodes_file, ux_file, uy_file):
    print("Chargement des données FreeFem...")
    try:
        points   = np.loadtxt(nodes_file)
        ux_data  = np.loadtxt(ux_file)
        uy_data  = np.loadtxt(uy_file)
        if len(points) != len(ux_data):
            raise ValueError("Nombre de nœuds différent du nombre de vitesses.")
        print("Création des interpolateurs...")
        ux_interp = CloughTocher2DInterpolator(points, ux_data)
        uy_interp = CloughTocher2DInterpolator(points, uy_data)
        print("Chargement terminé.")
        return ux_interp, uy_interp, points
    except Exception as e:
        print("Erreur lors du chargement :", e)
        return None, None, None


# ============================================================
# FONCTIONS PHYSIQUES
# ============================================================

def B_spire_3D(x, y, z, z0, R_coil, I):
    rho    = np.sqrt(x**2 + y**2)
    z_prime = z - z0
    if rho < 1e-12:
        return 0.0, 0.0, (mu0*I*R_coil**2) / (2*(R_coil**2 + z_prime**2)**1.5)
    r1_sq  = (R_coil - rho)**2 + z_prime**2
    r2_sq  = (R_coil + rho)**2 + z_prime**2
    k_sq   = 1 - r1_sq / r2_sq
    C      = mu0*I / (2*np.pi*np.sqrt(r2_sq))
    K      = ellipk(k_sq)
    E      = ellipe(k_sq)
    F      = (R_coil**2 + rho**2 + z_prime**2) / r1_sq
    B_rho  = C * (z_prime / rho) * (F*E - K)
    Bz     = C * (((R_coil**2 - rho**2 - z_prime**2)/r1_sq)*E + K)
    Bx     = B_rho * (x / rho)
    By     = B_rho * (y / rho)
    return Bx, By, Bz


def B_N_spires(x, y, z, N, I, R_coil, spacing):
    Bx_total = By_total = Bz_total = 0.0
    z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)
    for z0 in z_centers:
        bx, by, bz = B_spire_3D(x, y, z, z0, R_coil, I)
        Bx_total += bx
        By_total += by
        Bz_total += bz
    return Bx_total, By_total, Bz_total


def damping_y(x, y, Uy, charge_sign, xmur,
              gamma_max=1000.0, delta_zone=4e-4,
              delta_zone2=1e-7, delta_zone3=5e-4):
    damping = 0.0
    if y < delta_zone:
        if charge_sign > 0:
            damping = gamma_max * (delta_zone - y) / delta_zone
        else:
            return 0.0
    elif y > D - delta_zone:
        if charge_sign < 0:
            damping = gamma_max * (y - (D - delta_zone)) / delta_zone
        else:
            return 0.0
    walls = [lwall, D - lwall, lwall + eta, D - lwall - eta]
    for ywall in walls:
        if abs(y - ywall) < delta_zone2 and x >= xmur:
            damping = max(damping,
                          gamma_max * (delta_zone2 - abs(y - ywall)) / delta_zone2)
    return -damping * Uy


def s_ellipse_damping(X, U, xc, yc, a, b, theta_min, theta_max, eps=1e-7):
    dx  = X[0] - xc
    dy  = X[1] - yc
    val = (dx/a)**2 + (dy/b)**2
    if val <= 1:
        theta = np.arctan2(dy/b, dx/a)
        theta = np.clip(theta, theta_min, theta_max)
        ex = xc + a * np.cos(theta)
        ey = yc + b * np.sin(theta)
        nx = np.cos(theta) / a
        ny = np.sin(theta) / b
        norm = np.sqrt(nx*nx + ny*ny)
        nx /= norm; ny /= norm
        X[0] = ex + eps*nx
        X[1] = ey + eps*ny
        U[:] = 0.0
    return X, U


def get_exit_zone(y):
    if y <= lwall:
        return "bas (de Na+)"
    elif y >= D - lwall:
        return "haut (de Cl-)"
    else:
        return "milieu"


def simulate_until_exit(X0, U0, charge_sign, dt,
                        N, I, R_coil, spacing, z_coil,
                        L, xmur,
                        ux_interp=None, uy_interp=None,
                        max_steps=100):
    x_coil = L / 5
    y_coil = D / 2

    centers = {
        "BL": (xmur, lwall + Rtip),
        "BR": (xmur, lwall + Rtip),
        "TL": (xmur, D - lwall - eta + Rtip),
        "TR": (xmur, D - lwall - eta + Rtip),
    }
    angles = {
        "BR": (-np.pi/2, np.pi),
        "BL": (np.pi,     np.pi/2),
        "TR": (-np.pi/2, np.pi),
        "TL": (np.pi,     np.pi/2),
    }

    X = [X0.copy()]; U = [U0.copy()]
    i = 0

    while X[-1][0] < L and i < max_steps:
        x, y, z = X[-1]; ux, uy, uz = U[-1]

        Bx, By, Bz = B_N_spires(
            x - x_coil, y - y_coil, z - z_coil,
            N, I, R_coil, spacing
        )
        B = np.array([Bx, By, Bz]) / B0

        ux_flow = uy_flow = 0.0
        if ux_interp is not None:
            vx = ux_interp(x, y); vy = uy_interp(x, y)
            ux_flow = 0.0 if np.isnan(vx) else vx
            uy_flow = 0.0 if np.isnan(vy) else vy

        u_flow = np.array([ux_flow, uy_flow, 0.0])

        def f(u):
            return charge_sign * np.cross(u + u_flow, B)

        U_int = U[-1] + dt * f(U[-1])
        U_new = U[-1] + dt/2 * (f(U[-1]) + f(U_int))
        U_new[1] += dt * damping_y(x, y, U_new[1], charge_sign, xmur)

        X_new = X[-1] + dt * U_new

        for key in ["BL", "BR", "TL", "TR"]:
            xc, yc = centers[key]
            thmin, thmax = angles[key]
            X_new, U_new = s_ellipse_damping(
                X_new, U_new, xc, yc, delta, Rtip, thmin, thmax
            )

        walls = [lwall, D-lwall, lwall+eta, D-lwall-eta]
        if x >= xmur - delta:
            for ywall in walls:
                if (y - ywall) * (X_new[1] - ywall) < 0:
                    X_new[1] = ywall; U_new[1] = 0

        X.append(X_new); U.append(U_new); i += 1
        if X_new[1] < 0 or X_new[1] > D:
            break

    X = np.array(X); U = np.array(U)
    exit_point = X[-1]
    zone = get_exit_zone(exit_point[1])
    duration = i * dt
    return X, U, exit_point, zone, duration


def run_multi_lap(n_particles_by_charge, total_laps, dt,
                  N, I, R_coil, spacing, z_coil, L, xmur,
                  ux_interp=None, uy_interp=None,
                  verbose=False):
    particles = []
    for charge, count in n_particles_by_charge.items():
        sign  = 1 if charge == 'Na+' else (-1 if charge == 'Cl-' else 0)
        color = 'b' if charge == 'Na+' else ('r' if charge == 'Cl-' else 'g')
        for _ in range(count):
            y0 = np.random.uniform(0, D)
            particles.append({
                'charge_sign': sign, 'type': charge, 'couleur': color,
                'trajectoires': [], 'actif': True, 'laps_completed': 0,
                'initial_pos': np.array([0.0, y0, 0.0])
            })

    all_traj = []
    bilan_detail = []
    zones = ["bas (de Na+)", "milieu", "haut (de Cl-)"]
    types = ["Na+", "Cl-", "H20"]

    for lap in range(1, total_laps + 1):
        actives = [p for p in particles if p['actif']]
        stats_detail = {zone: {t: 0 for t in types} for zone in zones}

        for p in actives:
            X0 = p['reinject_pos'] if p['trajectoires'] else p['initial_pos']
            U0 = np.array([1.0, 0.0, 0.0])

            traj, _, exit_point, zone, duration = simulate_until_exit(
                X0, U0, p['charge_sign'], dt,
                N, I, R_coil, spacing, z_coil, L, xmur,
                ux_interp, uy_interp
            )
            p['trajectoires'].append(traj)
            all_traj.append((traj, p['couleur']))
            p['laps_completed'] += 1

            if zone in stats_detail:
                stats_detail[zone][p['type']] += 1

            if zone == "milieu":
                p['reinject_pos'] = np.array([0.0, exit_point[1], 0.0])
                p['actif'] = True
            else:
                p['actif'] = False

        bilan_detail.append(stats_detail)

        if not any(p['actif'] for p in particles):
            break

    return all_traj, particles, bilan_detail


# ============================================================
# DESSIN DU DOMAINE
# ============================================================

def draw_domain(L):
    Lwall = L / 8; xmur = L - Lwall
    n = 1.5; alpha = 2.0 / n
    plt.plot([0,L],[0,0],'k'); plt.plot([0,L],[D,D],'k')
    plt.plot([0,0],[0,D],'k')
    plt.plot([L,L],[0,lwall],'k')
    plt.plot([L,L],[lwall+eta, D-lwall-eta],'k')
    plt.plot([L,L],[D-lwall, D],'k')
    plt.plot([xmur,L],[lwall,lwall],'k')
    plt.plot([xmur,L],[D-lwall,D-lwall],'k')
    plt.plot([xmur,L],[lwall+eta,lwall+eta],'k')
    plt.plot([xmur,L],[D-lwall-eta,D-lwall-eta],'k')

    def tip_curve(xc, yc):
        theta = np.linspace(3*np.pi/2, np.pi/2, 300)
        ct = np.cos(theta); st = np.sin(theta)
        xp = np.sign(ct)*np.abs(ct)**alpha
        yp = np.sign(st)*np.abs(st)**alpha
        return xc + delta*xp, yc + Rtip*yp

    x_b, y_b = tip_curve(xmur, lwall + Rtip)
    plt.plot(x_b, y_b, 'k')
    x_t, y_t = tip_curve(xmur, D - lwall - eta + Rtip)
    plt.plot(x_t, y_t, 'k')
    plt.plot([L,L],[0, lwall+eta/2],'r--',lw=1)
    plt.plot([L,L],[lwall+eta/2, D-lwall-eta/2],'r--',lw=1)
    plt.plot([L,L],[D-lwall-eta/2, D],'r--',lw=1)


# ============================================================
# BOUCLE D'OPTIMISATION
# ============================================================

if __name__ == "__main__":

    # --- Paramètres de simulation fixes ---
    dt          = 1e-3
    n1, n2, n3  = 30, 30, 30          # Réduire pour aller plus vite
    total_laps  = 2
    n_particles = {'Na+': n2, 'Cl-': n3, 'H20': n1}

    # --- Chargement FreeFem (une seule fois) ---
    ux_interp, uy_interp, ff_points = load_freefem_data(
        'nodes.txt', 'ux.txt', 'uy.txt'
    )

    # --- Génération de toutes les combinaisons ---
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
  

    

    # --- Variables de suivi ---
    best_score  = -np.inf
    best_params = None
    best_bilan  = None

    def evaluate_combo(combo):
        params = dict(zip(keys, combo))
        L_val = params['L']
        xmur_val = L_val - L_val / 8
    
        # Mode “fast scan” : moins de particules et moins de tours
        n_particles_fast = {'Na+': 10, 'Cl-': 10, 'H20': 10}
        total_laps_fast = 1
    
        try:
            _, _, bilan_detail = run_multi_lap(
                n_particles_fast, total_laps_fast, dt,
                N       = params['N'],
                I       = params['I'],
                R_coil  = params['R_coil'],
                spacing = params['spacing'],
                z_coil  = params['z_coil'],
                L       = L_val,
                xmur    = xmur_val,
                ux_interp = ux_interp,
                uy_interp = uy_interp,
                verbose   = False
            )
            score = compute_score(bilan_detail)
        except Exception as e:
            print(f"Erreur pour {params} : {e}")
            score = -np.inf
    
        return (params, score)
    # --- Lancer le grid search multi-core ---
    print(f"\n{'='*60}")
    print(f"OPTIMISATION : {len(combos)} combinaisons à tester (multi-core)")
    print(f"{'='*60}\n")
    
    # --- Lancer le grid search multi-core ---
    print(f"\n{'='*60}")
    print(f"OPTIMISATION : {len(combos)} combinaisons à tester (multi-core)")
    print(f"{'='*60}\n")

    best_score  = -np.inf
    best_params = None
    best_bilan  = None
    
    # Lancer en parallèle
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_combo, combos))
    
    # --- Fichier CSV de résultats ---
    csv_file = "resultats_optimisation.csv"
    keys = list(PARAM_GRID.keys())
    fieldnames = keys + ['score', 'iteration']
    
    # Création du fichier et en-tête si le fichier n'existe pas
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    # Sauvegarde des résultats
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for idx, (params, score) in enumerate(results, start=1):
            row = {k: params[k] for k in keys}
            row['score'] = score
            row['iteration'] = idx
            writer.writerow(row)
    
            if score > best_score:
                best_score  = score
                best_params = params.copy()

    # ============================================================
    # RÉSULTAT FINAL
    # ============================================================

    print(f"\n{'='*60}")
    print("OPTIMISATION TERMINÉE")
    print(f"{'='*60}")
    print(f"Meilleur score    : {best_score}")
    print(f"Meilleurs paramètres :")
    for k, v in best_params.items():
        print(f"  {k} = {v}")

    # --- Tracé final avec les meilleurs paramètres ---
    print("\nRelance de la simulation avec les meilleurs paramètres pour tracé...")
    L_best   = best_params['L']
    xmur_best = L_best - L_best / 8

    all_traj_best, particles_best, _ = run_multi_lap(
        n_particles, total_laps, dt,
        N       = best_params['N'],
        I       = best_params['I'],
        R_coil  = best_params['R_coil'],
        spacing = best_params['spacing'],
        z_coil  = best_params['z_coil'],
        L       = L_best,
        xmur    = xmur_best,
        ux_interp = ux_interp,
        uy_interp = uy_interp,
        verbose   = True
    )

    max_lap = max(p['laps_completed'] for p in particles_best) if particles_best else 0
    for lap in range(1, max_lap + 1):
        plt.figure(figsize=(10, 5))
        draw_domain(L_best)
        #if ff_points is not None:
        #    plt.scatter(ff_points[:,0], ff_points[:,1],
        #                color='skyblue', s=10, label='Nœuds maillage')

        labels_done = {'Na+': False, 'Cl-': False, 'H20': False}
        for p in particles_best:
            if lap <= len(p['trajectoires']):
                traj  = p['trajectoires'][lap-1]
                label = None
                if not labels_done[p['type']]:
                    label = p['type']; labels_done[p['type']] = True
                plt.plot(traj[:,0], traj[:,1],
                         color=p['couleur'], lw=0.7, alpha=0.7, label=label)

        R_coil_best = best_params['R_coil']
        x_coil_best = L_best / 5
        theta_c = np.linspace(0, 2*np.pi, 300)
        plt.plot(x_coil_best + R_coil_best*np.cos(theta_c),
                 D/2          + R_coil_best*np.sin(theta_c),
                 'k', alpha=0.5, label="Bobine")

        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        plt.title(f"Tour {lap} — Meilleurs paramètres (score={best_score})")
        plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(); plt.tight_layout()
        
        plt.show()
