"""
Simulation multi-tours de particules dans le désalinisateur

Auteurs : bonaventure & audrey & thomas
Ce module implémente une simulation complète de trajectoires de particules
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.interpolate import CloughTocher2DInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMÈTRES GÉOMÉTRIQUES DU DOMAINE (constants)
# ─────────────────────────────────────────────────────────────────────────────

#L     = 0.076              # Longueur totale du domaine (m)
D     = 0.00276             # Hauteur totale du domaine (m)
lwall = D / 10             # Épaisseur caractéristique du mur latéral
Lwall = D                  # Longueur caractéristique du mur
eta   = D / 10             # Paramètre géométrique pour définir Rtip
Rtip  = eta / 2            # Rayon de l'extrémité (tip)
delta = 6 * Rtip           # Largeur de la zone d'influence autour du tip
#xmur  = L - Lwall          # Position du mur magnétique (début des pointes)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMÈTRES DES BOBINES — valeurs par défaut
# ─────────────────────────────────────────────────────────────────────────────

R_coil  = D / 2              # Rayon de la spire (m)
I       = 1.5                # Courant traversant la spire (A)
N       = 100                # Nombre total de spires
spacing = 0.001              # Espacement entre les spires (m)
mu0     = 4 * np.pi * 1e-7  # Perméabilité magnétique du vide (H/m)
x_coil  = 0.00948            # Position x de la bobine (m)
y_coil  = D / 2              # Position y de la bobine (m)
z_coil  = -0.054             # Position z de la bobine (m)
B0      = 200e-7             # Facteur de normalisation du champ (T)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHARGEMENT DES DONNÉES FREEFEM
# ─────────────────────────────────────────────────────────────────────────────

def load_freefem_data(nodes_file, ux_file, uy_file):
    """
    Charge les données FreeFem++ et construit des interpolateurs
    CloughTocher2D pour les composantes (ux, uy) du champ de vitesse.

    Retourne (ux_interp, uy_interp, points) ou (None, None, None).
    """
    print("Chargement des données FreeFem...")
    try:
        points  = np.loadtxt(nodes_file)
        ux_data = np.loadtxt(ux_file)
        uy_data = np.loadtxt(uy_file)
        if len(points) != len(ux_data):
            raise ValueError("Nombre de nœuds ≠ nombre de valeurs de vitesse.")
        print("Création des interpolateurs...")
        ux_interp = CloughTocher2DInterpolator(points, ux_data)
        uy_interp = CloughTocher2DInterpolator(points, uy_data)
        print("Chargement terminé.")
        return ux_interp, uy_interp, points
    except Exception as e:
        print("Erreur lors du chargement :", e)
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. FONCTIONS PHYSIQUES
# ─────────────────────────────────────────────────────────────────────────────

def B_spire_3D(x, y, z, I_val, R_val, z0=0.0):
    """
    Champ magnétique d'une spire circulaire de rayon R_val parcourue par
    un courant I_val, centrée en (0, 0, z0) dans le plan xy.

    Retourne (Bx, By, Bz).
    """
    rho     = np.sqrt(x**2 + y**2)
    z_prime = z - z0

    if rho < 1e-12:
        return 0.0, 0.0, (mu0 * I_val * R_val**2) / \
               (2 * (R_val**2 + z_prime**2)**1.5)

    r1_sq = (R_val - rho)**2 + z_prime**2
    r2_sq = (R_val + rho)**2 + z_prime**2
    k_sq  = 1.0 - r1_sq / r2_sq

    C = mu0 * I_val / (2 * np.pi * np.sqrt(r2_sq))
    K = ellipk(k_sq)
    E = ellipe(k_sq)

    F    = (R_val**2 + rho**2 + z_prime**2) / r1_sq
    B_rho = C * (z_prime / rho) * (F * E - K)
    Bz   = C * (((R_val**2 - rho**2 - z_prime**2) / r1_sq) * E + K)
    Bx   = B_rho * (x / rho)
    By   = B_rho * (y / rho)

    return Bx, By, Bz


def B_N_spires(x, y, z, I_val, R_val, N_spires, spac):
    """
    Champ magnétique total de N_spires spires empilées le long de z,
    séparées par un espacement `spac`.
    """
    Bx_total = By_total = Bz_total = 0.0
    z_centers = np.linspace(-spac * (N_spires - 1) / 2,
                             spac * (N_spires - 1) / 2,
                             N_spires)
    for z0 in z_centers:
        bx, by, bz = B_spire_3D(x, y, z, I_val, R_val, z0)
        Bx_total += bx
        By_total += by
        Bz_total += bz
    return Bx_total, By_total, Bz_total


def damping_y(x, y, Uy, D_val, charge_sign, xmur_val,
              gamma_max=1000.0,
              delta_zone=4e-4,
              delta_zone2=1e-7,
              lwall_val=0.0,
              eta_val=0.0):
    """
    Terme d'amortissement vertical simulant l'adhérence des ions aux
    électrodes (bas pour Na+, haut pour Cl-) et aux murs internes.
    """
    damping = 0.0
    # Électrode du bas (Na+)
    if y < delta_zone:
        if charge_sign > 0:
            damping = gamma_max * (delta_zone - y) / delta_zone
        else:
            return 0.0
    # Électrode du haut (Cl-)
    elif y > D_val - delta_zone:
        if charge_sign < 0:
            damping = gamma_max * (y - (D_val - delta_zone)) / delta_zone
        else:
            return 0.0
    # Parois internes (séparateurs)
    walls = [lwall_val, D_val - lwall_val,
             lwall_val + eta_val, D_val - lwall_val - eta_val]
    for ywall in walls:
        if abs(y - ywall) < delta_zone2 and x >= xmur_val:
            damping = max(damping,
                          gamma_max * (delta_zone2 - abs(y - ywall)) / delta_zone2)
    return -damping * Uy


def stick_and_slide_on_quarter_ellipse(X, U, xc, yc, a, b,
                                       theta_min, theta_max, eps=1e-7):
    """
    Si la particule pénètre dans un quart d'ellipse (pointe séparatrice),
    elle est projetée sur la surface et sa vitesse est annulée (adhérence).
    """
    dx  = X[0] - xc
    dy  = X[1] - yc
    val = (dx / a)**2 + (dy / b)**2

    if val <= 1:
        theta = np.arctan2(dy / b, dx / a)
        theta = np.clip(theta, theta_min, theta_max)

        ex = xc + a * np.cos(theta)
        ey = yc + b * np.sin(theta)

        nx = np.cos(theta) / a
        ny = np.sin(theta) / b
        norm = np.sqrt(nx**2 + ny**2)
        nx /= norm;  ny /= norm

        X[0] = ex + eps * nx
        X[1] = ey + eps * ny
        U[:] = 0.0   # adhérence complète sur la pointe

    return X, U


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMULATION D'UN TOUR JUSQU'À SORTIE
# ─────────────────────────────────────────────────────────────────────────────

def get_exit_zone(y):
    """
    Classifie la position de sortie en trois zones :
      'bas (de Na+)'  — électrode basse, capture Na+
      'haut (de Cl-)' — électrode haute, capture Cl-
      'milieu'        — canal central, particule réinjectée

    CORRECTION : la version précédente ne retournait rien (None) pour les
    particules dans la zone de tip (lwall < y < lwall+eta et symétrique),
    provoquant un crash en aval.
    """
    if y <= lwall:
        return "bas (de Na+)"
    elif y >= D - lwall:
        return "haut (de Cl-)"
    elif lwall + eta <= y <= D - lwall - eta:
        return "milieu"
    else:
        # Zone de tip (lwall < y < lwall+eta  ou  D-lwall-eta < y < D-lwall)
        # La particule n'a pas été proprement séparée → réinjection
        return "milieu"


def simulate_until_exit(X0, U0, charge_sign, dt,
                        ux_interp=None, uy_interp=None,
                        bobine=None, max_steps=1000):
    """
    Intègre la trajectoire d'une particule jusqu'à sortie (x ≥ L) ou
    jusqu'à max_steps itérations.

    La force de Lorentz est intégrée par un schéma RK2.
    Le champ de vitesse du fluide (FreeFem) est superposé à chaque pas.

    Retourne (X, U, exit_point, zone, duration).
    """
    # ── Extraction des paramètres bobine ─────────────────────────────────────
    L_loc       = bobine["L"]
    I_loc       = bobine["I"]
    R_loc       = bobine["Rayon"]
    N_loc       = bobine["Nb_spire"]
    spac_loc    = bobine["spacing"]
    xc_loc      = bobine["x_coil"]
    yc_loc      = bobine["y_coil"]
    zc_loc      = bobine["z_coil"]
    B0_loc      = bobine["B0"]
    xmur_loc    = L_loc - Lwall   # position du début du séparateur

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

        # ── Champ magnétique (paramètres depuis bobine) ───────────────────────
        Bx, By, Bz = B_N_spires(
            x - xc_loc, y - yc_loc, z - zc_loc,
            I_loc, R_loc, N_loc, spac_loc
        )
        B = np.array([Bx, By, Bz]) / B0_loc

        def f(u, B_vec):
            return charge_sign * np.cross(u, np.reshape(B_vec, (3,)))

        # ── Intégration RK2 (force de Lorentz) ────────────────────────────────
        U_int = U[-1] + dt * f(U[-1], B)
        U_new = U[-1] + dt / 2 * (f(U[-1], B) + f(U_int, B))

        # ── Amortissement vertical (xmur depuis bobine) ───────────────────────
        U_new[1] += dt * damping_y(
            x, y, U_new[1], D, charge_sign,
            xmur_val=xmur_loc,
            lwall_val=lwall,
            eta_val=eta
        )

        # ── Vitesse du fluide porteur (FreeFem) ───────────────────────────────
        if ux_interp is not None and uy_interp is not None:
            ux_flow = float(ux_interp(x, y))
            uy_flow = float(uy_interp(x, y))
            if np.isnan(ux_flow): ux_flow = 0.0
            if np.isnan(uy_flow): uy_flow = 0.0
        else:
            ux_flow = uy_flow = 0.0

        U_total = U_new + np.array([ux_flow, uy_flow, 0.0])
        X_new   = X[-1] + dt * U_total

        # ── Projection sur les pointes elliptiques ────────────────────────────
        for key in ["BL", "BR", "TL", "TR"]:
            xc, yc = centers[key]
            thmin, thmax = angles[key]
            X_new, U_new = stick_and_slide_on_quarter_ellipse(
                X_new, U_new, xc, yc, delta, Rtip, thmin, thmax
            )

        # ── Collision avec les murs internes verticaux ────────────────────────
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

    # BUG CORRIGÉ : bloc dupliqué supprimé, un seul return
    X          = np.array(X)
    U          = np.array(U)
    exit_point = X[-1]
    zone       = get_exit_zone(exit_point[1])
    duration   = i * dt
    return X, U, exit_point, zone, duration


# ─────────────────────────────────────────────────────────────────────────────
# 5b. POSITIONS INITIALES FIXES (déterministe)
# ─────────────────────────────────────────────────────────────────────────────

def make_fixed_y_positions(n_particles, margin_fraction=0.05):
    """
    Génère n_particles positions y régulièrement espacées dans [0, D] avec
    une marge `margin_fraction × D` par rapport aux parois.

    Exemple d'utilisation dans un optimiseur déterministe :
        fixed_y  = make_fixed_y_positions(50)
        fixed_pos = {'Na+': fixed_y, 'Cl-': fixed_y, 'H20': []}
        run_multi_lap(..., fixed_positions=fixed_pos)
    """
    margin = D * margin_fraction
    return np.linspace(margin, D - margin, n_particles)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SIMULATION MULTI-TOURS
# ─────────────────────────────────────────────────────────────────────────────

# BUG CORRIGÉ #7 : signature complète avec bobine, preloaded_interps,
#                   fixed_positions ; interpolateurs chargés depuis les fichiers
#                   si non pré-calculés ; support des positions fixes
def run_multi_lap(n_particles_by_charge, total_laps, dt, bobine,
                  verbose=True, preloaded_interps=None,
                  fixed_positions=None):
    """
    Simule plusieurs tours de particules dans le désalinisateur.

    Paramètres
    ----------
    n_particles_by_charge : dict  {'Na+': n2, 'Cl-': n3, 'H20': n1}
    total_laps            : int   tours maximum
    dt                    : float pas de temps (s)
    bobine                : dict  tous les paramètres de la bobine/tube
    verbose               : bool  affichage détaillé
    preloaded_interps     : tuple (ux_interp, uy_interp) pré-construits, ou None.
                            Si None, les fichiers FreeFem sont chargés depuis
                            le disque (pattern : nodes{L}.txt / ux{L}.txt / uy{L}.txt).
    fixed_positions       : dict  {'Na+': array_y, 'Cl-': array_y, 'H20': []}
                            Si fourni, positions initiales déterministes.
                            Utiliser make_fixed_y_positions() pour construire.

    Retour : (all_trajectories, particles, bilan_detail_par_tour)
    """
    L_loc    = bobine["L"]
    xmur_loc = L_loc - Lwall

    # ── Chargement (ou réutilisation) des interpolateurs ─────────────────────
    if preloaded_interps is not None:
        ux_interp, uy_interp = preloaded_interps
        points = None
    else:
        nodes_file = f"../Nodes/nodes{L_loc}.txt"
        ux_file    = f"../Uxs/ux{L_loc}.txt"
        uy_file    = f"../Uys/uy{L_loc}.txt"
        ux_interp, uy_interp, points = load_freefem_data(
            nodes_file, ux_file, uy_file
        )

    # ── Initialisation des particules ─────────────────────────────────────────
    particles = []
    for charge, count in n_particles_by_charge.items():
        if   charge == 'Na+': sign, color = +1, 'b'
        elif charge == 'Cl-': sign, color = -1, 'r'
        else:                  sign, color =  0, 'g'

        # Positions y : fixes si fourni, sinon aléatoires
        if fixed_positions is not None and charge in fixed_positions:
            y_arr = np.asarray(fixed_positions[charge])
            if len(y_arr) != count:
                idx   = np.round(np.linspace(0, len(y_arr) - 1, count)).astype(int)
                y_arr = y_arr[idx]
        else:
            y_arr = np.random.uniform(0, D, size=count)

        for y0 in y_arr:
            X0 = np.array([0.0, float(y0), 0.0])
            U0 = np.array([1.0, 0.0, 0.0])
            particles.append({
                'charge_sign'   : sign,
                'type'          : charge,
                'couleur'       : color,
                'trajectoires'  : [],
                'actif'         : True,
                'laps_completed': 0,
                'initial_pos'   : X0.copy(),
            })

    all_trajectories      = []
    bilan_detail_par_tour = []

    # ── Boucle sur les tours ──────────────────────────────────────────────────
    for lap in range(1, total_laps + 1):
        actives = [p for p in particles if p['actif']]
        n_start = len(actives)
        if verbose:
            print(f"\n--- Tour {lap} ---")
            print(f"Particules démarrant ce tour : {n_start}")

        zones  = ["bas (de Na+)", "milieu", "haut (de Cl-)"]
        types  = ["Na+", "Cl-", "H20"]
        stats_detail         = {zone: {t: 0 for t in types} for zone in zones}
        reinjectees_par_type = {t: 0 for t in types}

        for p in actives:
            X0 = p['reinject_pos'] if p['trajectoires'] else p['initial_pos']
            U0 = np.array([1.0, 0.0, 0.0])

            traj, _, exit_point, zone, duration = simulate_until_exit(
                X0, U0, p['charge_sign'], dt,
                ux_interp, uy_interp, bobine
            )
            p['trajectoires'].append(traj)
            p.setdefault('durations', []).append(duration)
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

        # Particules "perdues" (sorties du domaine en y)
        total_sorties = sum(sum(d.values()) for d in stats_detail.values())
        nb_perdues    = n_start - total_sorties

        bilan_detail_par_tour.append(stats_detail)

        if verbose:
            if actives:
                durations = [p['durations'][-1] for p in actives
                             if p.get('durations')]
                if durations:
                    print(f"Durée moyenne du tour {lap} : "
                          f"{np.mean(durations):.4f} s")
            print(f"Nombre de particules perdues : {nb_perdues}")
            print(f"\n===== BILAN DES PARTICULES (Tour {lap}) =====")
            total_tour = sum(sum(d.values()) for d in stats_detail.values())
            print(f"Nombre total ayant terminé ce tour : {total_tour}")
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
            else:
                print("\nAucune particule réinjectée ce tour.")

        if not any(p['actif'] for p in particles):
            if verbose:
                print("Plus de particules à réinjecter. Simulation arrêtée.")
            break

    if verbose:
        print("\n" + "=" * 60)
        print("BILAN FINAL GLOBAL")
        print("=" * 60)
        max_laps = max(p['laps_completed'] for p in particles) if particles else 0
        for k in range(1, max_laps + 1):
            count = sum(1 for p in particles if p['laps_completed'] >= k)
            print(f" Tour {k} : {count} particules")

    return all_trajectories, particles, bilan_detail_par_tour


# ─────────────────────────────────────────────────────────────────────────────
# 7. DESSIN DU DOMAINE
# ─────────────────────────────────────────────────────────────────────────────
"""
def draw_domain():
    n     = 1.5
    alpha = 2.0 / n
    Rtip_d = eta / 2

    # Murs extérieurs
    plt.plot([0, L], [0, 0], 'k')
    plt.plot([0, L], [D, D], 'k')
    plt.plot([0, 0], [0, D], 'k')
    plt.plot([L, L], [0, lwall], 'k')
    plt.plot([L, L], [lwall + eta, D - lwall - eta], 'k')
    plt.plot([L, L], [D - lwall, D], 'k')

    # Murs internes
    plt.plot([xmur, L], [lwall,         lwall],         'k')
    plt.plot([xmur, L], [D - lwall,     D - lwall],     'k')
    plt.plot([xmur, L], [lwall + eta,   lwall + eta],   'k')
    plt.plot([xmur, L], [D-lwall - eta, D-lwall - eta], 'k')

    def tip_curve(x_center, delta_d, Rtip_d, y_center, alpha_d, theta):
        ct = np.cos(theta);  st = np.sin(theta)
        xp = np.sign(ct) * np.abs(ct)**alpha_d
        yp = np.sign(st) * np.abs(st)**alpha_d
        return x_center + delta_d * xp, y_center + Rtip_d * yp

    theta = np.linspace(3 * np.pi / 2, np.pi / 2, 300)

    x_b, y_b = tip_curve(xmur, delta, Rtip_d, lwall + Rtip_d, alpha, theta)
    plt.plot(x_b, y_b, 'k')

    x_t, y_t = tip_curve(xmur, delta, Rtip_d,
                          D - lwall - eta + Rtip_d, alpha, theta)
    plt.plot(x_t, y_t, 'k')

    # Lignes de séparation des zones de sortie
    plt.plot([L, L], [0,               lwall + eta / 2],         'r--', lw=1)
    plt.plot([L, L], [lwall + eta / 2, D - lwall - eta / 2],     'r--', lw=1)
    plt.plot([L, L], [D - lwall - eta / 2, D],                   'r--', lw=1)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 8. PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    dt         = 1e-3
    n1         = 0
    n2         = 100
    n3         = 100
    total_laps = 2

    n_particles      = {'Na+': n2, 'Cl-': n3, 'H20': n1}
    TOTAL_PARTICULES = n1 + n2 + n3

    print("==============================")
    print("=== SIMULATION MULTI-TOURS ===")
    print("==============================")
    print(f"Nombre total de particules : {TOTAL_PARTICULES}")
    print(f"Nombre de tours maximum    : {total_laps}")

    L_sim = 0.076
    bobine_sim = {
        "L"        : L_sim,
        "I"        : I,
        "Rayon"    : R_coil,
        "Nb_spire" : N,
        "spacing"  : spacing,
        "x_coil"   : x_coil,
        "y_coil"   : y_coil,
        "z_coil"   : z_coil,
        "B0"       : B0,
    }

    nodes_file = f"nodes{L_sim}.txt"
    ux_file    = f"ux{L_sim}.txt"
    uy_file    = f"uy{L_sim}.txt"

    ux_interp, uy_interp, points = load_freefem_data(nodes_file, ux_file, uy_file)

    trajectories, particles_data, bilan_detail = run_multi_lap(
        n_particles, total_laps, dt, bobine_sim,
        verbose           = True,
        preloaded_interps = (ux_interp, uy_interp) if ux_interp is not None else None,
    )

    max_lap = max(p['laps_completed'] for p in particles_data) if particles_data else 0

    for lap in range(1, max_lap + 1):
        plt.figure(figsize=(10, 5))
        draw_domain()
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

        # Représentation de la bobine
        theta_c = np.linspace(0, 2 * np.pi, 300)
        plt.plot(bobine_sim["x_coil"] + bobine_sim["Rayon"] * np.cos(theta_c),
                 bobine_sim["y_coil"] + bobine_sim["Rayon"] * np.sin(theta_c),
                 'k', alpha=0.5, label="Bobine magnétique")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Trajectoires — Tour {lap}")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
