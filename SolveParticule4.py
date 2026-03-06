"""
Simulation multi-tours de particules

Auteurs : bonaventure & audrey & thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.interpolate import CloughTocher2DInterpolator

# ============================================================================
# 1. FONCTIONS DE CHARGEMENT DES DONNÉES FREEFEM
# ============================================================================
def load_freefem_data(nodes_file, ux_file, uy_file):
    """
    Charge les données issues d'un calcul FreeFem et construit des interpolateurs
    pour les champs de vitesse.

    Paramètres
    ----------
    nodes_file : str
        Chemin vers le fichier contenant les coordonnées des nœuds (x, y).
    ux_file : str
        Chemin vers le fichier contenant la composante horizontale de la vitesse.
    uy_file : str
        Chemin vers le fichier contenant la composante verticale de la vitesse.

    Retour
    ------
    ux_interp : CloughTocher2DInterpolator or None
        Interpolateur 2D pour la composante ux (None en cas d'erreur).
    uy_interp : CloughTocher2DInterpolator or None
        Interpolateur 2D pour la composante uy (None en cas d'erreur).
    points : ndarray or None
        Tableau des coordonnées des nœuds (None en cas d'erreur).
    """
    print("Chargement des données FreeFem...")
    try:
        points = np.loadtxt(nodes_file)
        ux_data = np.loadtxt(ux_file)
        uy_data = np.loadtxt(uy_file)
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

# ============================================================================
# 2. PARAMÈTRES GÉOMÉTRIQUES DU DOMAINE (constants)
# ============================================================================
L = 0.027              # Longueur totale du domaine (m)
D = 0.0036             # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = L / 8          # Longueur caractéristique du mur
eta = D / 10           # Paramètre géométrique pour définir Rtip
Rtip = eta / 2         # Rayon de l'extrémité (tip)
delta = 6 * Rtip       # Largeur de la zone d'influence autour du tip
xmur = L - Lwall       # Position du mur magnétique (début des pointes)

# ============================================================================
# 3. PARAMÈTRES DES BOBINES (champ magnétique)
# ============================================================================
R_coil = D / 3         # Rayon de la spire (m)
I = 1                  # Courant traversant la spire (A)
N = 100                # Nombre total de spires
spacing = 0.001        # Espacement entre les spires (m)
mu0 = 4 * np.pi * 1e-7 # Perméabilité magnétique du vide (H/m)
x_coil = L / 5         # Position horizontale de la bobine
z_coil = 0           # Position verticale de la bobine (dans le plan)
B0 = 200e-7            # Facteur de normalisation du champ (utilisé dans force de Lorentz)

# ============================================================================
# 4. FONCTIONS PHYSIQUES (champ magnétique, amortissement, projection)
# ============================================================================
def B_spire(x, z, z0=0.0):
    """
    Calcule le champ magnétique (Bx, Bz) produit par une spire circulaire.

    Paramètres
    ----------
    x : float
        Coordonnée horizontale du point d'observation.
    z : float
        Coordonnée verticale du point d'observation.
    z0 : float, optionnel
        Position verticale du centre de la spire (par défaut 0).

    Retour
    ------
    Bx, Bz : float
        Composantes horizontale et verticale du champ magnétique.
    """
    rho = np.abs(x)
    z_prime = z - z0
    if rho < 1e-12:
        Bx = 0.0
        Bz = (mu0 * I * R_coil**2) / (2 * (R_coil**2 + z_prime**2)**(3/2))
        return Bx, Bz
    r1_sq = (R_coil - rho)**2 + z_prime**2
    r2_sq = (R_coil + rho)**2 + z_prime**2
    k_sq = 1 - r1_sq / r2_sq
    C = mu0 * I / (2 * np.pi * np.sqrt(r2_sq))
    K = ellipk(k_sq)
    E = ellipe(k_sq)
    F = (R_coil**2 + rho**2 + z_prime**2) / r1_sq
    B_rho = C * (z_prime / rho) * (F * E - K)
    B_z = C * (((R_coil**2 - rho**2 - z_prime**2) / r1_sq) * E + K)
    Bx = B_rho * np.sign(x) if x != 0 else 0.0
    return Bx, B_z

def B_N_spires(x, z):
    """
    Calcule le champ magnétique total produit par N spires identiques alignées verticalement.

    Paramètres
    ----------
    x, z : float
        Coordonnées du point d'observation.

    Retour
    ------
    Bx_total, Bz_total : float
        Composantes horizontale et verticale du champ total.
    """
    Bx_total, Bz_total = 0.0, 0.0
    z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)
    for z_center in z_centers:
        bx, bz = B_spire(x, z, z_center)
        Bx_total += bx
        Bz_total += bz
    return Bx_total, Bz_total

def damping_y(x, y, Uy, D, charge_sign,
              gamma_max=1000.0,
              delta_zone=4e-4,
              delta_zone2=1e-7,
              delta_zone3=5e-4,
              lwall=0.0,
              eta=0.0):
    """
    Amortissement appliqué à la composante verticale de la vitesse au voisinage des parois.

    Paramètres
    ----------
    x, y : float
        Position de la particule.
    Uy : float
        Composante verticale de la vitesse.
    D : float
        Hauteur totale du domaine.
    charge_sign : int
        Signe de la charge (-1, 0, 1).
    gamma_max, delta_zone, delta_zone2, delta_zone3, lwall, eta : float
        Paramètres de l'amortissement.

    Retour
    ------
    float
        Terme d'amortissement à ajouter à la dérivée de Uy (ou 0 si pas d'amortissement).
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
    # Parois internes (indépendant du signe)
    walls = [lwall, D - lwall, lwall + eta, D - lwall - eta]
    for ywall in walls:
        if abs(y - ywall) < delta_zone2 and x >= xmur:
            damping = max(damping,
                          gamma_max * (delta_zone2 - abs(y - ywall)) / delta_zone2)
    return -damping * Uy

def stick_and_slide_on_quarter_ellipse(X, U, xc, yc, a, b, theta_min, theta_max, eps=1e-5):
    """
    Projette la particule sur le bord extérieur d'un quart d'ellipse et
    ne conserve que la composante tangentielle de la vitesse.

    Paramètres
    ----------
    X : array_like de taille 3
        Position (x, y, z) de la particule.
    U : array_like de taille 3
        Vitesse (ux, uy, uz) de la particule.
    xc, yc : float
        Coordonnées du centre de l'ellipse.
    a, b : float
        Demi‑axes de l'ellipse (respectivement horizontal et vertical).
    theta_min, theta_max : float
        Angles (en radians) délimitant le quart d'ellipse considéré.
    eps : float, optionnel
        Petit décalage vers l'extérieur pour éviter d'être exactement sur le bord.

    Retour
    ------
    X, U : ndarray
        Position et vitesse corrigées.
    """
    dx = X[0] - xc
    dy = X[1] - yc
    val = (dx/a)**2 + (dy/b)**2
    if val <= 1:
        theta = np.arctan2(dy / b, dx / a)
        # Clamp dans l'intervalle du quart d'ellipse
        if theta < theta_min:
            theta = theta_min
        elif theta > theta_max:
            theta = theta_max
        # Point exact sur l'ellipse
        ex = xc + a * np.cos(theta)
        ey = yc + b * np.sin(theta)
        # Normale extérieure
        nx = np.cos(theta) / a
        ny = np.sin(theta) / b
        norm = np.sqrt(nx*nx + ny*ny)
        nx /= norm
        ny /= norm
        # Légèrement à l'extérieur
        X[0] = ex + eps * nx
        X[1] = ey + eps * ny
        # Vecteur tangent
        tx = -a * np.sin(theta)
        ty =  b * np.cos(theta)
        t_norm = np.sqrt(tx*tx + ty*ty)
        tx /= t_norm
        ty /= t_norm
        # Projection de la vitesse sur la tangente
        vt = U[0]*tx + U[1]*ty
        U[0] = vt * tx
        U[1] = vt * ty
    return X, U

# ============================================================================
# 5. SIMULATION D'UN TOUR JUSQU'À SORTIE
# ============================================================================
def get_exit_zone(y):
    """
    Détermine la zone de sortie en fonction de la coordonnée verticale y.

    Paramètres
    ----------
    y : float
        Coordonnée y du point de sortie.

    Retour
    ------
    str
        'bas (de Na+)', 'milieu' ou 'haut (de Cl-')'.
    """
    if y <= lwall + eta/2:
        return "bas (de Na+)"
    elif y >= D - lwall - eta/2:
        return "haut (de Cl-)"
    else:
        return "milieu"

def simulate_until_exit(X0, U0, charge_sign, dt, ux_interp=None, uy_interp=None, max_steps=100):
    """
    Simule la trajectoire d'une particule depuis une position initiale
    jusqu'à ce qu'elle sorte du domaine (x >= L) ou dépasse les bornes en y.

    Paramètres
    ----------
    X0 : array_like de taille 3
        Position initiale (x, y, z).
    U0 : array_like de taille 3
        Vitesse initiale (ux, uy, uz).
    charge_sign : int
        Signe de la charge (-1, 0, 1).
    dt : float
        Pas de temps de la simulation.
    max_steps : int, optionnel
        Nombre maximal de pas pour éviter les boucles infinies.

    Retour
    ------
    X : ndarray de shape (n_steps, 3)
        Trajectoire complète.
    U : ndarray de shape (n_steps, 3)
        Vitesses correspondantes.
    exit_point : ndarray de taille 3
        Dernier point de la trajectoire.
    zone : str
        Zone de sortie.
    """
    # Définition des quarts d'ellipse (centres et angles)
    centers = {
        "BL": (xmur, lwall + Rtip),
        "BR": (xmur, lwall + Rtip),
        "TL": (xmur, D - lwall - eta + Rtip),
        "TR": (xmur, D - lwall - eta + Rtip)
    }
    angles = {
        "BR": (-np.pi/2, np.pi),
        "BL": (np.pi, np.pi/2),
        "TR": (-np.pi/2, np.pi),
        "TL": (np.pi, np.pi/2)
    }

    X = [X0.copy()]
    U = [U0.copy()]
    i = 0

    while X[-1][0] < L and i < max_steps:
        x, y, z = X[-1]
        ux, uy, uz = U[-1]

        # Champ magnétique
        Bx, Bz = B_N_spires(x - x_coil, z - z_coil)
        B = np.array([Bx, 0.0, Bz]) / B0

        def f(u):
            return charge_sign * np.cross(u, B)

        # RK2 pour la vitesse
        U_int = U[-1] + dt * f(U[-1])
        U_new = U[-1] + dt/2 * (f(U[-1]) + f(U_int))

        # Amortissement vertical
        U_new[1] += dt * damping_y(
            x, y, U_new[1],
            D, charge_sign,
            lwall=lwall,
            eta=eta
        )

        # =========================
        # AJOUT DU CHAMP DE VITESSE FREEFEM
        # =========================
        
        if ux_interp is not None and uy_interp is not None:
            ux_flow = ux_interp(x, y)
            uy_flow = uy_interp(x, y)
        
            if np.isnan(ux_flow): ux_flow = 0
            if np.isnan(uy_flow): uy_flow = 0
        else:
            ux_flow = 0
            uy_flow = 0
        
        # vitesse totale = particule + fluide
        U_total = U_new + np.array([ux_flow, uy_flow, 0.0])
        
        # Position mise à jour
        X_new = X[-1] + dt * U_total

        # Projection sur les quarts d'ellipse
        for key in ["BL", "BR", "TL", "TR"]:
            xc, yc = centers[key]
            thmin, thmax = angles[key]
            X_new, U_new = stick_and_slide_on_quarter_ellipse(
                X_new, U_new,
                xc, yc,
                delta, Rtip,
                thmin, thmax
            )

        # Collisions avec les murs internes verticaux
        walls = [lwall, D-lwall, lwall+eta, D-lwall-eta]
        if x >= xmur - delta:
            for ywall in walls:
                if (y - ywall) * (X_new[1] - ywall) < 0:
                    X_new[1] = ywall
                    U_new[1] = 0

        X.append(X_new)
        U.append(U_new)
        i += 1

        # Arrêt si la particule sort du domaine en y
        if X_new[1] < 0 or X_new[1] > D:
            break

    X = np.array(X)
    U = np.array(U)
    exit_point = X[-1]
    zone = get_exit_zone(exit_point[1])
    return X, U, exit_point, zone

# ============================================================================
# 6. SIMULATION MULTI-TOURS AVEC BILAN DÉTAILLÉ
# ============================================================================
def run_multi_lap(n_particles_by_charge, total_laps, dt, verbose=True):
    """
    Lance la simulation multi-tours pour un ensemble de particules.

    Paramètres
    ----------
    n_particles_by_charge : dict
        Dictionnaire avec les clés 'Na+', 'Cl-', 'H20' et le nombre correspondant.
    total_laps : int
        Nombre maximum de tours à simuler.
    dt : float
        Pas de temps pour la simulation.
    verbose : bool, optionnel
        Si True, affiche le bilan détaillé après chaque tour.

    Retour
    ------
    all_trajectories : list of tuples (traj, color)
        Liste de toutes les trajectoires (pour le tracé) avec leur couleur.
    particles : list of dict
        Liste contenant les informations de chaque particule.
    bilan_detail_par_tour : list of dict
        Statistiques détaillées pour chaque tour (par zone et par type).
    """
    # Initialisation des particules pour le premier tour
    particles = []
    for charge, count in n_particles_by_charge.items():
        if charge == 'Na+':
            sign = 1
            color = 'b'
        elif charge == 'Cl-':
            sign = -1
            color = 'r'
        else:  # H20
            sign = 0
            color = 'g'
        for _ in range(count):
            y0 = np.random.uniform(0, D)
            X0 = np.array([0.0, y0, 0.0])
            U0 = np.array([1.0, 0.0, 0.0])
            particles.append({
                'charge_sign': sign,
                'type': charge,      # 'Na+', 'Cl-', 'H20'
                'couleur': color,
                'trajectoires': [],   # liste des trajectoires de chaque tour
                'actif': True,
                'laps_completed': 0,
                'initial_pos': X0.copy()
            })

    all_trajectories = []
    bilan_detail_par_tour = []  # pour stocker les stats détaillées par tour

    for lap in range(1, total_laps + 1):
        print(f"\n--- Tour {lap} ---")
        actives = [p for p in particles if p['actif']]
        n_start = len(actives)
        print(f"Particules démarrant ce tour : {n_start}")

        # Initialisation des compteurs pour ce tour
        zones = ["bas (de Na+)", "milieu", "haut (de Cl-)"]
        types = ["Na+", "Cl-", "H20"]
        stats_detail = {zone: {t: 0 for t in types} for zone in zones}
        reinjectees_par_type = {t: 0 for t in types}

        for p in actives:
            # Déterminer la position de départ pour ce tour
            if p['trajectoires']:
                X0 = p['reinject_pos']
                U0 = np.array([1.0, 0.0, 0.0])
            else:
                X0 = p['initial_pos']
                U0 = np.array([1.0, 0.0, 0.0])

            # Simulation jusqu'à sortie
            traj, _, exit_point, zone = simulate_until_exit(
                X0, U0, p['charge_sign'], dt,
                ux_interp, uy_interp
            )
            p['trajectoires'].append(traj)
            all_trajectories.append((traj, p['couleur']))

            p['laps_completed'] += 1

            # Mise à jour des compteurs
            if zone in stats_detail:
                stats_detail[zone][p['type']] += 1

            if zone == "milieu":
                reinjectees_par_type[p['type']] += 1
                p['reinject_pos'] = np.array([0.0, exit_point[1], 0.0])
                p['actif'] = True
            else:
                p['actif'] = False

        bilan_detail_par_tour.append(stats_detail)

        # Affichage du bilan détaillé pour ce tour
        if verbose:
            print("\n===== BILAN DES PARTICULES (Tour {}) =====".format(lap))
            total_particules_tour = sum(sum(d.values()) for d in stats_detail.values())
            print(f"Nombre total de particules ayant terminé ce tour : {total_particules_tour}")
            for zone in zones:
                total_zone = sum(stats_detail[zone].values())
                if total_zone > 0:
                    print(f"\nZone {zone} (total {total_zone}) :")
                    for t in types:
                        nb = stats_detail[zone][t]
                        pct = nb / total_zone * 100
                        print(f"  {t} : {nb} ({pct:.1f}%)")
                else:
                    print(f"\nZone {zone} : aucune particule")
            # Affichage des proportions parmi les réinjectées (sorties milieu)
            total_reinject = sum(reinjectees_par_type.values())
            if total_reinject > 0:
                print("\nParmi les particules réinjectées (sorties milieu) :")
                for t in types:
                    nb = reinjectees_par_type[t]
                    pct = nb / total_reinject * 100
                    print(f"  {t} : {nb} ({pct:.1f}%)")
            else:
                print("\nAucune particule réinjectée ce tour.")

        # Vérification si plus de particules actives
        if not any(p['actif'] for p in particles):
            print("Plus de particules à réinjecter. Simulation arrêtée.")
            break

    # Bilan final global
    if verbose:
        print("\n" + "=" * 60)
        print("BILAN FINAL GLOBAL")
        print("=" * 60)
        max_laps = max(p['laps_completed'] for p in particles) if particles else 0
        print("Répartition par nombre de tours effectués :")
        for k in range(1, max_laps + 1):
            count = sum(1 for p in particles if p['laps_completed'] >= k)
            print(f" Tour {k}  : {count} particules")

    return all_trajectories, particles, bilan_detail_par_tour

# ============================================================================
# 7. FONCTION DE DESSIN DU DOMAINE
# ============================================================================
def draw_domain():
    """
    Dessine le domaine avec les parois, les pointes et les lignes de séparation.
    """
    plt.plot([0, L], [0, 0], 'k')
    plt.plot([0, L], [D, D], 'k')
    plt.plot([0, 0], [0, D], 'k')
    plt.plot([L, L], [0, lwall], 'k')
    plt.plot([L, L], [lwall+eta, D-lwall-eta], 'k')
    plt.plot([L, L], [D-lwall, D], 'k')
    plt.plot([xmur, L], [lwall, lwall], 'k')
    plt.plot([xmur, L], [D-lwall, D-lwall], 'k')
    plt.plot([xmur, L], [lwall+eta, lwall+eta], 'k')
    plt.plot([xmur, L], [D-lwall-eta, D-lwall-eta], 'k')
    theta = np.linspace(3*np.pi/2, np.pi/2, 300)
    x_tip = delta * np.cos(theta) + xmur
    y_tip_bottom = Rtip * np.sin(theta) + lwall + Rtip
    y_tip_top = Rtip * np.sin(theta) + D - lwall - eta + Rtip
    plt.plot(x_tip, y_tip_bottom, 'k')
    plt.plot(x_tip, y_tip_top, 'k')
    plt.plot([xmur-delta, xmur-delta], [0, lwall + eta/2], 'k--', linewidth=1)
    plt.plot([xmur-delta, xmur-delta], [lwall + eta/2, D - lwall - eta/2], 'k--', linewidth=1)
    plt.plot([xmur-delta, xmur-delta], [D - lwall - eta/2, D], 'k--', linewidth=1)

# ============================================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    # -------------------------------------------------
    # Paramètres de simulation (à modifier au besoin)
    # -------------------------------------------------
    dt = 1e-3                     # Pas de temps (s)
    n1 = 50                        # Nombre de particules neutres (H20)
    n2 = 50                        # Nombre de particules positives (Na+)
    n3 = 50                        # Nombre de particules négatives (Cl-)
    total_laps = 2                 # Nombre maximum de tours à simuler

    n_particles = {'Na+': n2, 'Cl-': n3, 'H20': n1}
    TOTAL_PARTICULES = n1 + n2 + n3
    print(f"==============================")
    print(f"=== SIMULATION MULTI-TOURS ===")
    print(f"==============================")
    print(f"Nombre total de particules : {TOTAL_PARTICULES}")
    print(f"Nombre de tours maximum : {total_laps}")
    # -------------------------------------------------
    # Chargement des données FreeFem (si disponibles: les fichiers doivent être dans le même dossier)
    # -------------------------------------------------
    ux_interp, uy_interp, points = load_freefem_data('nodes.txt', 'ux.txt', 'uy.txt')

    
    # -------------------------------------------------
    # Lancement de la simulation
    # -------------------------------------------------
    trajectories, particles_data, bilan_detail = run_multi_lap(n_particles, total_laps, dt, verbose=True)

   
    # -------------------------------------------------
    # Tracé global de toutes les trajectoires (tous tours confondus)
    # -------------------------------------------------
    #plt.figure(figsize=(10, 5))
    #draw_domain()

    #if points is not None:
    #    plt.scatter(points[:, 0], points[:, 1], color='skyblue', s=10, label='Nœuds du maillage')


    # -------------------------------------------------
    # Tracé par tour (un graphique distinct pour chaque tour)
    # -------------------------------------------------
    max_lap = max(p['laps_completed'] for p in particles_data) if particles_data else 0
    for lap in range(1, max_lap + 1):
        plt.figure(figsize=(10, 5))
        draw_domain()
        if points is not None:
            plt.scatter(points[:, 0], points[:, 1], color='skyblue', s=10)

        # Tracé des trajectoires de ce tour
        for p in particles_data:
            if lap <= len(p['trajectoires']):
                traj = p['trajectoires'][lap-1]
                plt.plot(traj[:, 0], traj[:, 1], color=p['couleur'], linewidth=0.7, alpha=0.7)

        # Tracé de la bobine (optionnel)
        theta = np.linspace(0, 2*np.pi, 300)
        x_circ = x_coil + R_coil * np.cos(theta)
        y_circ = D/2 + R_coil * np.sin(theta)
        plt.plot(x_circ, y_circ, 'r', alpha=0.5)

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Trajectoires du tour {lap}")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()