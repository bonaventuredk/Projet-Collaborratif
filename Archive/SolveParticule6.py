"""
Simulation multi-tours de particules dans le désalinisateur

Auteurs : bonaventure & audrey & thomas

Corrections appliquées :
  - Bug fix : B_N_spires() utilisait un espacement hardcodé (0.001 m) au lieu
    du paramètre `spacing` passé en argument.
  - run_multi_lap() accepte maintenant preloaded_interps=(ux_interp, uy_interp)
    pour éviter de recharger les fichiers FreeFem à chaque appel (gain majeur
    lors des optimisations).
  - max_steps de simulate_until_exit passe à 1000 (100 était trop court pour
    les particules fortement déviées).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.interpolate import CloughTocher2DInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMÈTRES GÉOMÉTRIQUES DU DOMAINE (constants)
# ─────────────────────────────────────────────────────────────────────────────

L = 0.027              # Longueur totale du domaine (m)
L = 0.076
D = 0.019              # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = D              # Longueur caractéristique du mur
eta = D / 10           # Paramètre géométrique pour définir Rtip
Rtip = eta / 2         # Rayon de l'extrémité (tip)
delta = 6 * Rtip       # Largeur de la zone d'influence autour du tip
xmur = L - Lwall       # Position du mur magnétique (début des pointes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMÈTRES DES BOBINES (champ magnétique)
# ─────────────────────────────────────────────────────────────────────────────

R_coil  = D / 1000            # Rayon de la spire (m)
#R_coil  = 0.03636
I       = 1                  # Courant traversant la spire (A)
I       = 3.808 
N       = 100                # Nombre total de spires
N       = 369
spacing = 0.001              # Espacement entre les spires (m)
spacing = 0.003005
mu0     = 4 * np.pi * 1e-7  # Perméabilité magnétique du vide (H/m)
x_coil  = L / 5             # Position suivant x de la bobine
x_coil  = 0.00948 
y_coil  = D / 2             # Position suivant y de la bobine
z_coil  = -0.0054                  # Position suivant z de la bobine
B0      = 200e-7             # Facteur de normalisation du champ
B0      = 4.208e-08


# ─────────────────────────────────────────────────────────────────────────────
# 3. FONCTIONS DE CHARGEMENT DES DONNÉES FREEFEM
# ─────────────────────────────────────────────────────────────────────────────

def load_freefem_data(nodes_file, ux_file, uy_file):
    """
    Charge les données de FreeFem et construit des interpolateurs pour les
    vitesses (ux, uy).

    Paramètres
    ----------
    nodes_file : str  — chemin vers le fichier de coordonnées (x, y)
    ux_file    : str  — composante x de la vitesse
    uy_file    : str  — composante y de la vitesse

    Retour
    ------
    (ux_interp, uy_interp, points) ou (None, None, None) en cas d'erreur.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. FONCTIONS PHYSIQUES (champ magnétique, amortissement, projection)
# ─────────────────────────────────────────────────────────────────────────────

def B_spire_3D(x, y, z, I, R_coil, z0=0.0):
    """
    Champ magnétique d'une spire circulaire de rayon R_coil centrée en
    (0, 0, z0) dans le plan xy.

    Retour : (Bx, By, Bz)
    """
    rho = np.sqrt(x * x + y * y)
    z_prime = z - z0

    if rho < 1e-12:
        return 0.0, 0.0, (mu0 * I * R_coil ** 2) / (2 * (R_coil ** 2 + z_prime ** 2) ** (3 / 2))

    r1_sq = (R_coil - rho) ** 2 + z_prime ** 2
    r2_sq = (R_coil + rho) ** 2 + z_prime ** 2
    k_sq  = 1 - r1_sq / r2_sq

    C = mu0 * I / (2 * np.pi * np.sqrt(r2_sq))
    K = ellipk(k_sq)
    E = ellipe(k_sq)

    F = (R_coil ** 2 + rho ** 2 + z_prime ** 2) / r1_sq

    B_rho = C * (z_prime / rho) * (F * E - K)
    Bz    = C * (((R_coil ** 2 - rho ** 2 - z_prime ** 2) / r1_sq) * E + K)
    Bx    = B_rho * (x / rho)
    By    = B_rho * (y / rho)

    return Bx, By, Bz


def B_N_spires(x, y, z, I, R_coil, nb_spires, spacing):
    """
    Champ magnétique total de nb_spires spires empilées le long de z,
    séparées d'un espacement `spacing`.

    CORRECTION : le paramètre `spacing` est désormais utilisé (la version
    précédente l'ignorait et utilisait 0.001 m en dur).
    """
    Bx_total = 0.0
    By_total = 0.0
    Bz_total = 0.0

    N_loc = nb_spires
    # Positions des centres de chaque spire le long de z
    z_centers = np.linspace(-spacing * (N_loc - 1) / 2,
                             spacing * (N_loc - 1) / 2,
                             N_loc)

    for z0 in z_centers:
        bx, by, bz = B_spire_3D(x, y, z, I, R_coil, z0)
        Bx_total += bx
        By_total += by
        Bz_total += bz

    return Bx_total, By_total, Bz_total


def damping_y(x, y, Uy, D, charge_sign, xmur,
              gamma_max=1000.0,
              delta_zone=4e-4,
              delta_zone2=1e-7,
              delta_zone3=5e-4,
              lwall=0.0,
              eta=0.0):
    """
    Terme d'amortissement vertical près des parois pour simuler l'arrêt des
    ions contre les électrodes / murs séparateurs.
    """
    damping = 0.0
    # Mur du bas
    if y < delta_zone:
        if charge_sign > 0:
            damping = gamma_max * (delta_zone - y) / delta_zone
        else:
            return 0.0
    # Mur du haut
    elif y > D - delta_zone:
        if charge_sign < 0:
            damping = gamma_max * (y - (D - delta_zone)) / delta_zone
        else:
            return 0.0
    # Parois internes
    walls = [lwall, D - lwall, lwall + eta, D - lwall - eta]
    for ywall in walls:
        if abs(y - ywall) < delta_zone2 and x >= xmur:
            damping = max(damping,
                          gamma_max * (delta_zone2 - abs(y - ywall)) / delta_zone2)
    return -damping * Uy


def stick_and_slide_on_quarter_ellipse(X, U, xc, yc, a, b,
                                       theta_min, theta_max, eps=1e-5):
    """
    Projette la particule sur un quart d'ellipse si elle en franchit la
    frontière, et conserve uniquement la composante tangentielle de la vitesse.
    """
    dx  = X[0] - xc
    dy  = X[1] - yc
    val = (dx / a) ** 2 + (dy / b) ** 2
    if val <= 1:
        theta = np.arctan2(dy / b, dx / a)
        theta = np.clip(theta, theta_min, theta_max)
        ex = xc + a * np.cos(theta)
        ey = yc + b * np.sin(theta)
        nx = np.cos(theta) / a
        ny = np.sin(theta) / b
        norm = np.sqrt(nx * nx + ny * ny)
        nx /= norm;  ny /= norm
        X[0] = ex + eps * nx
        X[1] = ey + eps * ny
        tx = -a * np.sin(theta)
        ty  =  b * np.cos(theta)
        t_norm = np.sqrt(tx * tx + ty * ty)
        tx /= t_norm;  ty /= t_norm
        vt   = U[0] * tx + U[1] * ty
        U[0] = vt * tx
        U[1] = vt * ty
    return X, U


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMULATION D'UN TOUR JUSQU'À SORTIE
# ─────────────────────────────────────────────────────────────────────────────

def get_exit_zone(y):
    """
    Classe la position de sortie y en trois zones :
      'bas (de Na+)'   — électrode captant les cations Na+
      'haut (de Cl-)'  — électrode captant les anions Cl-
      'milieu'         — eau non séparée (réinjection)
    """
    if y <= lwall + eta / 2:
        return "bas (de Na+)"
    elif y >= D - lwall - eta / 2:
        return "haut (de Cl-)"
    else:
        return "milieu"


def simulate_until_exit(X0, U0, charge_sign, dt,
                        ux_interp=None, uy_interp=None,
                        bobine=None, max_steps=1000):
    """
    Intègre la trajectoire d'une particule jusqu'à ce qu'elle quitte le
    domaine (x ≥ L) ou dépasse max_steps itérations.

    L'intégration de la force de Lorentz utilise un schéma RK2.
    La vitesse du fluide (champ FreeFem) est ajoutée à chaque pas.

    Retour : (X, U, exit_point, zone)
    """
    L_loc  = bobine["L"]
    xmur_loc = L_loc - Lwall

    centers = {
        "BL": (xmur_loc, lwall + Rtip),
        "BR": (xmur_loc, lwall + Rtip),
        "TL": (xmur_loc, D - lwall - eta + Rtip),
        "TR": (xmur_loc, D - lwall - eta + Rtip),
    }
    angles = {
        "BR": (-np.pi / 2,  np.pi),
        "BL": ( np.pi,      np.pi / 2),
        "TR": (-np.pi / 2,  np.pi),
        "TL": ( np.pi,      np.pi / 2),
    }

    X = [X0.copy()]
    U = [U0.copy()]
    i = 0

    while X[-1][0] < L_loc and i < max_steps:
        x, y, z = X[-1]
        ux, uy, uz = U[-1]

        # Récupération des paramètres bobine
        x_coil_loc  = bobine["x_coil"]
        y_coil_loc  = bobine["y_coil"]
        z_coil_loc  = bobine["z_coil"]
        I_loc       = bobine["I"]
        R_coil_loc  = bobine["Rayon"]
        nb_spires   = bobine["Nb_spire"]
        spacing_loc = bobine["spacing"]
        B0_loc      = bobine["B0"]

        Bx, By, Bz = B_N_spires(
            x - x_coil_loc,
            y - y_coil_loc,
            z - z_coil_loc,
            I_loc, R_coil_loc, nb_spires, spacing_loc
        )
        B = np.array([Bx, By, Bz]) / B0_loc

        def f(u, B_vec):
            B_vec = np.reshape(B_vec, (3,))
            return charge_sign * np.cross(u, B_vec)

        # Intégration RK2 (force de Lorentz)
        U_int = U[-1] + dt * f(U[-1], B)
        U_new = U[-1] + dt / 2 * (f(U[-1], B) + f(U_int, B))

        # Amortissement vertical près des parois
        U_new[1] += dt * damping_y(
            x, y, U_new[1], D, charge_sign,
            xmur=xmur_loc, lwall=lwall, eta=eta
        )

        # Champ de vitesse FreeFem (fluide porteur)
        if ux_interp is not None and uy_interp is not None:
            ux_flow = ux_interp(x, y)
            uy_flow = uy_interp(x, y)
            if np.isnan(ux_flow): ux_flow = 0.0
            if np.isnan(uy_flow): uy_flow = 0.0
        else:
            ux_flow = 0.0
            uy_flow = 0.0

        U_total = U_new + np.array([ux_flow, uy_flow, 0.0])
        X_new   = X[-1] + dt * U_total

        # Projection sur les quarts d'ellipse des pointes
        for key in ["BL", "BR", "TL", "TR"]:
            xc, yc = centers[key]
            thmin, thmax = angles[key]
            X_new, U_new = stick_and_slide_on_quarter_ellipse(
                X_new, U_new, xc, yc, delta, Rtip, thmin, thmax
            )

        # Collisions avec les murs internes verticaux
        walls = [lwall, D - lwall, lwall + eta, D - lwall - eta]
        if x >= xmur_loc - delta:
            for ywall in walls:
                if (y - ywall) * (X_new[1] - ywall) < 0:
                    X_new[1] = ywall
                    U_new[1] = 0.0

        X.append(X_new)
        U.append(U_new)
        i += 1

        if X_new[1] < 0 or X_new[1] > D:
            break

    X = np.array(X)
    U = np.array(U)
    exit_point = X[-1]
    zone = get_exit_zone(exit_point[1])
    return X, U, exit_point, zone


# ─────────────────────────────────────────────────────────────────────────────
# 6. SIMULATION MULTI-TOURS AVEC BILAN
# ─────────────────────────────────────────────────────────────────────────────

def make_fixed_y_positions(n_particles, margin_fraction=0.05):
    """
    Calcule n_particles positions y régulièrement espacées dans [0, D],
    en laissant une marge relative `margin_fraction` par rapport aux parois.

    Utilisation typique dans l'optimisation déterministe :
        fixed_y = make_fixed_y_positions(50)
        fixed_pos = {'Na+': fixed_y, 'Cl-': fixed_y, 'H20': []}
        run_multi_lap(..., fixed_positions=fixed_pos)
    """
    margin = D * margin_fraction
    return np.linspace(margin, D - margin, n_particles)


def run_multi_lap(n_particles_by_charge, total_laps, dt, bobine,
                  verbose=True, preloaded_interps=None,
                  fixed_positions=None):
    """
    Simule plusieurs tours de particules dans le désalinisateur.

    Paramètres
    ----------
    n_particles_by_charge : dict  {'Na+': n2, 'Cl-': n3, 'H20': n1}
    total_laps            : int   nombre maximum de tours
    dt                    : float pas de temps (s)
    bobine                : dict  paramètres de la bobine
    verbose               : bool  affichage détaillé
    preloaded_interps     : tuple (ux_interp, uy_interp) pré-calculés, ou None.
                            Si fourni, les fichiers FreeFem ne sont PAS rechargés
                            (gain important lors des optimisations répétées).
    fixed_positions       : dict  {'Na+': array_of_y, 'Cl-': array_of_y, 'H20': []}
                            Si fourni, les positions initiales en y sont fixes
                            (déterministe, pas de tirage aléatoire).
                            Utiliser make_fixed_y_positions() pour construire ce dict.

    Retour
    ------
    (all_trajectories, particles, bilan_detail_par_tour)
    """
    # ── Initialisation des particules ────────────────────────────────────────
    particles = []
    for charge, count in n_particles_by_charge.items():
        if   charge == 'Na+': sign, color = +1, 'b'
        elif charge == 'Cl-': sign, color = -1, 'r'
        else:                  sign, color =  0, 'g'

        # Positions y : fixes si fourni, aléatoires sinon
        if fixed_positions is not None and charge in fixed_positions:
            y_positions = np.asarray(fixed_positions[charge])
            # Adapter au nombre de particules demandé (rééchantillonnage si besoin)
            if len(y_positions) != count:
                indices  = np.round(np.linspace(0, len(y_positions) - 1, count)).astype(int)
                y_positions = y_positions[indices]
        else:
            y_positions = np.random.uniform(0, D, size=count)

        for y0 in y_positions:
            X0 = np.array([0.0, float(y0), 0.0])
            U0 = np.array([1.0, 0.0, 0.0])
            particles.append({
                'charge_sign'  : sign,
                'type'         : charge,
                'couleur'      : color,
                'trajectoires' : [],
                'actif'        : True,
                'laps_completed': 0,
                'initial_pos'  : X0.copy(),
            })

    all_trajectories      = []
    bilan_detail_par_tour = []

    L_loc  = bobine["L"]
    xmur_loc = L_loc - Lwall

    # ── Chargement (ou réutilisation) des interpolateurs FreeFem ─────────────
    if preloaded_interps is not None:
        ux_interp, uy_interp = preloaded_interps
        points = None
    else:
        filenameNodes = "nodes" + str(L_loc) + ".txt"
        filenameUx    = "ux"    + str(L_loc) + ".txt"
        filenameUy    = "uy"    + str(L_loc) + ".txt"
        ux_interp, uy_interp, points = load_freefem_data(
            filenameNodes, filenameUx, filenameUy
        )

    # ── Boucle sur les tours ─────────────────────────────────────────────────
    for lap in range(1, total_laps + 1):
        if verbose:
            print(f"\n--- Tour {lap} ---")
        actives = [p for p in particles if p['actif']]
        if verbose:
            print(f"Particules démarrant ce tour : {len(actives)}")

        zones  = ["bas (de Na+)", "milieu", "haut (de Cl-)"]
        types  = ["Na+", "Cl-", "H20"]
        stats_detail        = {zone: {t: 0 for t in types} for zone in zones}
        reinjectees_par_type = {t: 0 for t in types}

        for p in actives:
            X0 = p['reinject_pos'] if p['trajectoires'] else p['initial_pos']
            U0 = np.array([1.0, 0.0, 0.0])

            traj, _, exit_point, zone = simulate_until_exit(
                X0, U0, p['charge_sign'], dt,
                ux_interp, uy_interp, bobine
            )
            p['trajectoires'].append(traj)
            all_trajectories.append((traj, p['couleur']))
            p['laps_completed'] += 1

            if zone in stats_detail:
                stats_detail[zone][p['type']] += 1

            if zone == "milieu":
                reinjectees_par_type[p['type']] += 1
                p['reinject_pos'] = np.array([0.0, exit_point[1], 0.0])
                p['actif'] = True
            else:
                p['actif'] = False

        bilan_detail_par_tour.append(stats_detail)

        # ── Affichage du bilan ───────────────────────────────────────────────
        if verbose:
            print(f"\n===== BILAN DES PARTICULES (Tour {lap}) =====")
            total_particules_tour = sum(sum(d.values()) for d in stats_detail.values())
            print(f"Nombre total : {total_particules_tour}")
            for zone in zones:
                total_zone = sum(stats_detail[zone].values())
                if total_zone > 0:
                    print(f"\nZone {zone} (total {total_zone}) :")
                    for t in types:
                        nb  = stats_detail[zone][t]
                        pct = nb / total_zone * 100
                        print(f"  {t} : {nb} ({pct:.1f}%)")
                else:
                    print(f"\nZone {zone} : aucune particule")
            total_reinject = sum(reinjectees_par_type.values())
            if total_reinject > 0:
                print("\nParmi les particules réinjectées (sorties milieu) :")
                for t in types:
                    nb  = reinjectees_par_type[t]
                    pct = nb / total_reinject * 100
                    print(f"  {t} : {nb} ({pct:.1f}%)")

        if not any(p['actif'] for p in particles):
            if verbose:
                print("Plus de particules à réinjecter. Simulation arrêtée.")
            break

    # ── Bilan final global ───────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("BILAN FINAL GLOBAL")
        print("=" * 60)
        max_laps = max(p['laps_completed'] for p in particles) if particles else 0
        print("Répartition par nombre de tours effectués :")
        for k in range(1, max_laps + 1):
            count = sum(1 for p in particles if p['laps_completed'] >= k)
            print(f" Tour {k} : {count} particules")

    return all_trajectories, particles, bilan_detail_par_tour


# ─────────────────────────────────────────────────────────────────────────────
# 7. DESSIN DU DOMAINE
# ─────────────────────────────────────────────────────────────────────────────

def draw_domain(L_draw, xmur_draw, D_draw=D, lwall_draw=lwall,
                eta_draw=eta, Rtip_draw=Rtip):
    """Dessine le contour du domaine (tube + pointes séparatrices)."""
    plt.plot([0, L_draw],      [0, 0],             'k')
    plt.plot([0, L_draw],      [D_draw, D_draw],   'k')
    plt.plot([0, 0],           [0, D_draw],         'k')
    plt.plot([L_draw, L_draw], [0, lwall_draw],     'k')
    plt.plot([L_draw, L_draw], [lwall_draw + eta_draw, D_draw - lwall_draw - eta_draw], 'k')
    plt.plot([L_draw, L_draw], [D_draw - lwall_draw, D_draw], 'k')
    plt.plot([xmur_draw, L_draw], [lwall_draw,             lwall_draw],             'k')
    plt.plot([xmur_draw, L_draw], [D_draw - lwall_draw,    D_draw - lwall_draw],   'k')
    plt.plot([xmur_draw, L_draw], [lwall_draw + eta_draw,  lwall_draw + eta_draw], 'k')
    plt.plot([xmur_draw, L_draw], [D_draw - lwall_draw - eta_draw,
                                    D_draw - lwall_draw - eta_draw], 'k')
    delta_draw = 6 * Rtip_draw
    theta      = np.linspace(3 * np.pi / 2, np.pi / 2, 300)
    x_tip      = delta_draw * np.cos(theta) + xmur_draw
    plt.plot(x_tip, Rtip_draw * np.sin(theta) + lwall_draw + Rtip_draw, 'k')
    plt.plot(x_tip, Rtip_draw * np.sin(theta) + D_draw - lwall_draw - eta_draw + Rtip_draw, 'k')
    plt.plot([xmur_draw - delta_draw, xmur_draw - delta_draw], [0, lwall_draw + eta_draw / 2],
             'k--', linewidth=1)
    plt.plot([xmur_draw - delta_draw, xmur_draw - delta_draw],
             [lwall_draw + eta_draw / 2, D_draw - lwall_draw - eta_draw / 2],
             'k--', linewidth=1)
    plt.plot([xmur_draw - delta_draw, xmur_draw - delta_draw],
             [D_draw - lwall_draw - eta_draw / 2, D_draw], 'k--', linewidth=1)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    dt        = 1e-3
    n1        = 0
    n2        = 100
    n3        = 100
    total_laps = 2

    n_particles    = {'Na+': n2, 'Cl-': n3, 'H20': n1}
    TOTAL_PARTICULES = n1 + n2 + n3

    print("==============================")
    print("=== SIMULATION MULTI-TOURS ===")
    print("==============================")
    print(f"Nombre total de particules : {TOTAL_PARTICULES}")
    print(f"Nombre de tours maximum    : {total_laps}")

    ux_interp, uy_interp, points = load_freefem_data(
        'nodes.txt', 'ux.txt', 'uy.txt'
    )

    L_sim = 0.076
    #bobine = {
    #    "L"        : L_sim,
    #    "I"        : 1,
    #    "Rayon"    : D / 2,
    #    "Nb_spire" : N,
    #    "spacing"  : 0.001,
    #    "x_coil"   : L_sim / 5,
    #    "y_coil"   : D / 2,
    #    "z_coil"   : 0.0,
    #    "B0"       : 200e-7,
    #}

    #### Configuration Bobine 
    bobine = {
        "L"        : L_sim,
        "I"        : 1,
        "Rayon"    : D / 3,
        "Nb_spire" : 100,
        "spacing"  : 0.001,
        "x_coil"   : 0.0,
        "y_coil"   : D / 2,
        "z_coil"   : -D - 0.0002,
        "B0"       : 200e-7,
    }

    trajectories, particles_data, bilan_detail = run_multi_lap(
        n_particles, total_laps, dt, bobine=bobine,
        verbose=True,
        preloaded_interps=(ux_interp, uy_interp) if ux_interp is not None else None
    )

    xmur_sim  = L_sim - Lwall
    max_lap   = max(p['laps_completed'] for p in particles_data) if particles_data else 0

    for lap in range(1, max_lap + 1):
        plt.figure(figsize=(10, 5))
        draw_domain(L_sim, xmur_sim)
        if points is not None:
            plt.scatter(points[:, 0], points[:, 1],
                        color='skyblue', s=10, label='Nœuds maillage')

        labels_done = {'Na+': False, 'Cl-': False, 'H20': False}
        for p in particles_data:
            if lap <= len(p['trajectoires']):
                traj  = p['trajectoires'][lap - 1]
                label = None
                if not labels_done[p['type']]:
                    label = p['type']
                    labels_done[p['type']] = True
                plt.plot(traj[:, 0], traj[:, 1],
                         color=p['couleur'], linewidth=0.7,
                         alpha=0.7, label=label)

        theta  = np.linspace(0, 2 * np.pi, 300)
        x_circ = bobine["x_coil"] + bobine["Rayon"] * np.cos(theta)
        y_circ = bobine["y_coil"] + bobine["Rayon"] * np.sin(theta)
        plt.plot(x_circ, y_circ, 'k', alpha=0.5, label="Bobine magnétique")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Trajectoires — Tour {lap}")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
