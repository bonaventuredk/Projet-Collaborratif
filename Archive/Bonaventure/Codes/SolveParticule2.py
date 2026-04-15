"""
Simulation multi-tours de particules dans le déssalinisateur

Version : APRES OPTIMISATION 

Auteurs : bonaventure & audrey & thomas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.interpolate import CloughTocher2DInterpolator



# 1. PARAMÈTRES GÉOMÉTRIQUES DU DOMAINE (constants)

L = 0.027              # Longueur totale du domaine (m) (fixée initialement pour les tests mais sera optimisée plus tard)
D = 0.00276             # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = L / 8          # Longueur caractéristique du mur
eta = D / 10           # Paramètre géométrique pour définir Rtip
Rtip = eta / 2         # Rayon de l'extrémité (tip)
delta = 6 * Rtip       # Largeur de la zone d'influence autour du tip
xmur = L - Lwall       # Position du mur magnétique (début des pointes)


# 2. PARAMÈTRES DES BOBINES (champ magnétique)

R_coil = 0.002         # Rayon de la spire (m)
I = 0.005                  # Courant traversant la spire (A)
N = 40                # Nombre total de spires
spacing = 0.0005        # Espacement entre les spires (m)
mu0 = 4 * np.pi * 1e-7 # Perméabilité magnétique du vide (H/m)
x_coil = L / 5         # Position suivant x de la bobine            <----
y_coil = D / 2         # Position suivant y de la bobine            <----
z_coil = 0             # Position suivant z de la bobine            <----
B0 = 3e-7           # Facteur de normalisation du champ (utilisé dans la force de Lorentz)



# 3. FONCTIONS DE CHARGEMENT DES DONNÉES FREEFEM

def load_freefem_data(nodes_file, ux_file, uy_file):
    """
    Cette fonction charge les données de FreeFem et fait des interpolateurs pour les vitesses.

    Elle prend trois fichiers : un pour les nœuds (x,y), un pour ux, un pour uy.

    Elle retourne les interpolateurs pour ux et uy, et les points. Si y'a une erreur, ça retourne None.
    """
    print("Chargement des données FreeFem...")
    try:
        points = np.loadtxt(nodes_file)  # On charge les coordonnées des points
        ux_data = np.loadtxt(ux_file)  # Composante x de la vitesse
        uy_data = np.loadtxt(uy_file)  # Composante y de la vitesse
        if len(points) != len(ux_data):
            raise ValueError("Nombre de nœuds différent du nombre de vitesses.")  # Vérif qu'on a le bon nombre
        print("Création des interpolateurs...")
        ux_interp = CloughTocher2DInterpolator(points, ux_data)  # Interpolateur pour ux
        uy_interp = CloughTocher2DInterpolator(points, uy_data)  # Interpolateur pour uy
        print("Chargement terminé.")
        return ux_interp, uy_interp, points
    except Exception as e:
        print("Erreur lors du chargement :", e)  # Si ça plante, on dit pourquoi
        return None, None, None


# 4. FONCTIONS PHYSIQUES (champ magnétique, amortissement, projection)

def B_spire_3D(x, y, z, z0=0.0):
    """
    Calcule le champ magnétique d'une spire circulaire.
    La spire est au centre (0,0,z0) dans le plan x-y.
    """
    rho = np.sqrt(x**2 + y**2)  # Distance radiale
    z_prime = z - z0  # Distance en z par rapport au centre

    if rho < 1e-12:  # Si on est au centre, éviter division par zéro
        return 0.0, 0.0, (mu0*I*R_coil**2)/(2*(R_coil**2+z_prime**2)**(3/2))

    r1_sq = (R_coil-rho)**2 + z_prime**2
    r2_sq = (R_coil+rho)**2 + z_prime**2

    k_sq = 1 - r1_sq/r2_sq  # Paramètre pour les intégrales elliptiques

    C = mu0*I/(2*np.pi*np.sqrt(r2_sq))  # Constante

    K = ellipk(k_sq)  # Intégrale elliptique complète de première espèce
    E = ellipe(k_sq)  # Intégrale elliptique complète de deuxième espèce

    F = (R_coil**2 + rho**2 + z_prime**2)/r1_sq

    B_rho = C*(z_prime/rho)*(F*E - K)  # Composante radiale

    Bz = C*(((R_coil**2 - rho**2 - z_prime**2)/r1_sq)*E + K)  # Composante z

    Bx = B_rho*(x/rho)  # Composante x
    By = B_rho*(y/rho)  # Composante y

    return Bx, By, Bz

def B_N_spires(x, y, z):
    """
    Champ total de N spires empilées le long de z.
    """
    Bx_total = 0.0
    By_total = 0.0
    Bz_total = 0.0

    z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)  # Positions des centres des spires

    for z0 in z_centers:  # Pour chaque spire
        bx, by, bz = B_spire_3D(x, y, z, z0)  # Champ de cette spire

        Bx_total += bx  # Additionner
        By_total += by
        Bz_total += bz

    return Bx_total, By_total, Bz_total

def damping_y(x, y, Uy, D, charge_sign,
              gamma_max=1000.0,
              delta_zone=4e-4,
              delta_zone2=1e-7,
              delta_zone3=5e-4,
              lwall=0.0,
              eta=0.0):
    """
    Amortissement pour la vitesse verticale près des parois.

    Prend la position, la vitesse Uy, etc., et retourne le terme d'amortissement.
    """
    damping = 0.0
    # Mur du bas
    if y < delta_zone:
        if charge_sign > 0:  # Seulement pour les positives
            damping = gamma_max * (delta_zone - y) / delta_zone
        else:
            return 0.0
    # Mur du haut
    elif y > D - delta_zone:
        if charge_sign < 0:  # Seulement pour les négatives
            damping = gamma_max * (y - (D - delta_zone)) / delta_zone
        else:
            return 0.0
    # Parois internes (pour tous)
    walls = [lwall, D - lwall, lwall + eta, D - lwall - eta]
    for ywall in walls:
        if abs(y - ywall) < delta_zone2 and x >= xmur:
            damping = max(damping,
                          gamma_max * (delta_zone2 - abs(y - ywall)) / delta_zone2)
    return -damping * Uy

def s_ellipse_damping(X, U, xc, yc, a, b, theta_min, theta_max, eps=1e-7):
    """
    Projette la particule sur un quart d'ellipse et impose la condition d'adhérence (vitesse nulle).

    Paramètres:
        X : array [x, y]  position de la particule (modifiée)
        U : array [ux, uy] vitesse de la particule (modifiée)
        xc, yc : centre de l'ellipse
        a, b : demi-axes de l'ellipse
        theta_min, theta_max : angles de début et fin du quart d'ellipse (en radians)
        eps : petite distance pour placer la particule juste à l'extérieur de l'ellipse

    Retourne:
        X, U après projection et adhérence
    """
    dx = X[0] - xc
    dy = X[1] - yc
    val = (dx / a)**2 + (dy / b)**2

    if val <= 1:  # Particule à l'intérieur ou sur la pointe
        theta = np.arctan2(dy / b, dx / a)  # angle paramétrique

        # Clamp dans l'intervalle autorisé
        if theta < theta_min:
            theta = theta_min
        elif theta > theta_max:
            theta = theta_max

        # Point sur l'ellipse
        ex = xc + a * np.cos(theta)
        ey = yc + b * np.sin(theta)

        # Normale extérieure
        nx = np.cos(theta) / a
        ny = np.sin(theta) / b
        norm = np.sqrt(nx * nx + ny * ny)
        nx /= norm
        ny /= norm

        # Placer la particule juste à l'extérieur de la pointe
        X[0] = ex + eps * nx
        X[1] = ey + eps * ny

        # Condition de non-glissement : la particule colle, vitesse nulle
        U[:] = 0.0

    return X, U


# 5. SIMULATION D'UN TOUR JUSQU'À SORTIE

def get_exit_zone(y):
    """
    Détermine la zone de sortie selon y.

    Retourne 'bas (de Na+)', 'milieu' ou 'haut (de Cl-)'.
    """
    if y <= lwall:
        return "bas (de Na+)"
    elif y >= D - lwall:
        return "haut (de Cl-)"
    elif y <= D - lwall - eta and y >= lwall + eta :
        return "milieu"

def simulate_until_exit(X0, U0, charge_sign, dt, ux_interp=None, uy_interp=None, max_steps=100):
    """
    Simule une particule jusqu'à ce qu'elle sorte (x >= L).

    Prend position initiale, vitesse, signe de charge, pas de temps, interpolateurs optionnels.

    Retourne trajectoire, vitesses, point de sortie, zone.
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
        Bx, By, Bz = B_N_spires(
            x - x_coil,
            y - y_coil,
            z - z_coil
        )
        
        B = np.array([Bx, By, Bz]) / B0

        # Champ de vitesse fluide
        if ux_interp is not None and uy_interp is not None:
            ux_flow = ux_interp(x, y)
            uy_flow = uy_interp(x, y)
        
            if np.isnan(ux_flow): ux_flow = 0
            if np.isnan(uy_flow): uy_flow = 0
        else:
            ux_flow = 0
            uy_flow = 0
        
        u_flow = np.array([ux_flow, uy_flow, 0.0])
        
        def f(u):
            # vitesse totale utilisée dans la force de Lorentz
            return charge_sign * np.cross(u + u_flow, B)
        
        # Schéma de Heun
        U_int = U[-1] + dt * f(U[-1])
        U_new = U[-1] + dt/2 * (f(U[-1]) + f(U_int))

        # Amortissement vertical
        U_new[1] += dt * damping_y(
            x, y, U_new[1],
            D, charge_sign,
            lwall=lwall,
            eta=eta
        )

        
        # AJOUT DU CHAMP DE VITESSE FREEFEM
        
        
        if ux_interp is not None and uy_interp is not None:
            ux_flow = ux_interp(x, y)
            uy_flow = uy_interp(x, y)
        
            if np.isnan(ux_flow): ux_flow = 0  # Si pas de valeur, mettre 0
            if np.isnan(uy_flow): uy_flow = 0
        else:
            ux_flow = 0
            uy_flow = 0
        
        
        # Position mise à jour
        X_new = X[-1] + dt * U_new

        # Projection sur les quarts d'ellipse
        for key in ["BL", "BR", "TL", "TR"]:
            xc, yc = centers[key]
            thmin, thmax = angles[key]
            X_new, U_new = s_ellipse_damping(
                X_new, U_new,
                xc, yc,
                delta, Rtip,
                thmin, thmax
            )

        # Collisions avec les murs internes verticaux
        walls = [lwall, D-lwall, lwall+eta, D-lwall-eta]
        if x >= xmur - delta:
            for ywall in walls:
                if (y - ywall) * (X_new[1] - ywall) < 0:  # Si traverse le mur
                    X_new[1] = ywall
                    U_new[1] = 0  # Vitesse verticale à 0

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
    X = np.array(X)
    U = np.array(U)
    exit_point = X[-1]
    zone = get_exit_zone(exit_point[1])
    duration = i * dt  # <-- ici on calcule la durée du tour
    return X, U, exit_point, zone, duration
    return X, U, exit_point, zone


# 6. SIMULATION MULTI-TOURS AVEC BILAN 

def run_multi_lap(n_particles_by_charge, total_laps, dt, verbose=True):
    """
    Lance la simulation multi-tours pour plusieurs particules.

    Prend dict des particules par type, nombre de tours, dt, verbose.

    Retourne trajectoires, données particules, bilan par tour.
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
            y0 = np.random.uniform(0, D)  # Position y aléatoire
            X0 = np.array([0.0, y0, 0.0])
            U0 = np.array([1.0, 0.0, 0.0])  # Vitesse initiale vers la droite
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
        actives = [p for p in particles if p['actif']]  # Particules encore actives
        n_start = len(actives)
        print(f"Particules démarrant ce tour : {n_start}")

        # Initialisation des compteurs pour ce tour
        zones = ["bas (de Na+)", "milieu", "haut (de Cl-)"]
        types = ["Na+", "Cl-", "H20"]
        stats_detail = {zone: {t: 0 for t in types} for zone in zones}
        reinjectees_par_type = {t: 0 for t in types}

        for p in actives:
            # Déterminer la position de départ pour ce tour
            if p['trajectoires']:  # Si pas le premier tour
                X0 = p['reinject_pos']
                U0 = np.array([1.0, 0.0, 0.0])
            else:
                X0 = p['initial_pos']
                U0 = np.array([1.0, 0.0, 0.0])

            # Simulation jusqu'à sortie
            traj, _, exit_point, zone, duration = simulate_until_exit(
                X0, U0, p['charge_sign'], dt,
                ux_interp, uy_interp
            )
            p['trajectoires'].append(traj)
            p.setdefault('durations', []).append(duration)  # <-- ici on stocke la durée
            #p['trajectoires'].append(traj)
            all_trajectories.append((traj, p['couleur']))

            p['laps_completed'] += 1

            # Mise à jour des compteurs
            if zone in stats_detail:
                stats_detail[zone][p['type']] += 1

            if zone == "milieu":  # Réinjecter si milieu
                reinjectees_par_type[p['type']] += 1
                p['reinject_pos'] = np.array([0.0, exit_point[1], 0.0])
                p['actif'] = True
            else:
                p['actif'] = False  # Sortie définitive
        if verbose:
            total_duration = np.mean([p['durations'][-1] for p in actives])
            print(f"Durée moyenne du tour {lap} : {total_duration:.4f} s")
                
        # Calcul des particules perdues
        total_sorties = sum(sum(d.values()) for d in stats_detail.values())
        nb_perdues = n_start - total_sorties  # Particules ayant touché un obstacle
        print(f"Nombre de particules perdues  : {nb_perdues}")
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


# 7. FONCTION DE DESSIN DU DOMAINE
def draw_domain():
    """
    Dessine le domaine avec les murs
    """
    # ===================================================
    # Paramètres géométriques (à ajuster si nécessaire)
    # ===================================================
    D = 0.00276            # Largeur du domaine (3.6 mm)
    L = 0.027             # Longueur totale
    lwall = D / 10        # Hauteur des murs
    Lwall = L / 8         # Position horizontale du mur (depuis la gauche)
    eta = D / 10          # Épaisseur du mur
    R = eta / 2           # Rayon de base de la pointe
    delta = 6 * R         # Profondeur de la pointe
    xmur = L - Lwall      # Position horizontale du début du mur (côté droit)
    n = 1.5               # Paramètre de forme de la pointe
    alpha = 2.0 / n       # Exposant pour la transformation

    # ===================================================
    # Dessin des murs extérieurs et des lignes horizontales
    # ===================================================
    plt.plot([0, L], [0, 0], 'k')                     # Bas
    plt.plot([0, L], [D, D], 'k')                     # Haut
    plt.plot([0, 0], [0, D], 'k')                     # Gauche
    plt.plot([L, L], [0, lwall], 'k')                 # Droite bas
    plt.plot([L, L], [lwall + eta, D - lwall - eta], 'k')  # Droite milieu
    plt.plot([L, L], [D - lwall, D], 'k')             # Droite haut

    # Murs internes horizontaux
    plt.plot([xmur, L], [lwall, lwall], 'k')          # Mur interne bas
    plt.plot([xmur, L], [D - lwall, D - lwall], 'k')  # Mur interne haut
    plt.plot([xmur, L], [lwall + eta, lwall + eta], 'k')      # Arête du bas
    plt.plot([xmur, L], [D - lwall - eta, D - lwall - eta], 'k')  # Arête du haut

    # ===================================================
    # Fonction pour générer une pointe
    # ===================================================
    def tip_curve(x_center, delta, R, y_center, alpha, theta):
        """
        Calcule les coordonnées (x, y) des points d'une pointe.
        Utilise la transformation : xp = sign(cosθ) * |cosθ|^α, yp = sign(sinθ) * |sinθ|^α.
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        xp = np.sign(ct) * np.abs(ct) ** alpha
        yp = np.sign(st) * np.abs(st) ** alpha
        x = x_center + delta * xp
        y = y_center + R * yp
        return x, y

    # Angles paramétriques : de 270° à 90° (3π/2 à π/2)
    theta = np.linspace(3 * np.pi / 2, np.pi / 2, 300)

    # ===================================================
    # Pointe inférieure
    # ===================================================
    x_tip_bottom, y_tip_bottom = tip_curve(xmur, delta, R, lwall + R, alpha, theta)
    plt.plot(x_tip_bottom, y_tip_bottom, 'k')

    # ===================================================
    # Pointe supérieure
    # ===================================================
    x_tip_top, y_tip_top = tip_curve(xmur, delta, R, D - lwall - eta + R, alpha, theta)
    plt.plot(x_tip_top, y_tip_top, 'k')

    # ===================================================
    # Lignes de séparation en pointillés
    # ===================================================
    plt.plot([L, L], [0, lwall + eta / 2], 'r--', linewidth=1)
    plt.plot([L, L], [lwall + eta / 2, D - lwall - eta / 2], 'r--', linewidth=1)
    plt.plot([L,L], [D - lwall - eta / 2, D], 'r--', linewidth=1)

# 8. PROGRAMME PRINCIPAL

if __name__ == "__main__":
    
    # Paramètres de simulation (à modifier au besoin)
    
    dt = 1e-3                     # Pas de temps (s)
    n1 = 100                       # Nombre de particules neutres (H20)
    n2 = 100                        # Nombre de particules positives (Na+)
    n3 = 100                        # Nombre de particules négatives (Cl-)
    total_laps = 2                 # Nombre maximum de tours à simuler

    n_particles = {'Na+': n2, 'Cl-': n3, 'H20': n1}
    TOTAL_PARTICULES = n1 + n2 + n3
    print(f"==============================")
    print(f"=== SIMULATION MULTI-TOURS ===")
    print(f"==============================")
    print(f"Nombre total de particules : {TOTAL_PARTICULES}")
    print(f"Nombre de tours maximum : {total_laps}")
    
    # Chargement des données FreeFem (si disponibles: les fichiers doivent être dans le même dossier)
   
    ux_interp, uy_interp, points = load_freefem_data('nodes.txt', 'ux.txt', 'uy.txt')

    
    
    # Lancement de la simulation
   
    trajectories, particles_data, bilan_detail = run_multi_lap(n_particles, total_laps, dt, verbose=True)

    # Tracé par tour (un graphique distinct pour chaque tour)

    max_lap = max(p['laps_completed'] for p in particles_data) if particles_data else 0
    
    for lap in range(1, max_lap + 1):
        plt.figure(figsize=(10, 5))
        draw_domain()
        if points is not None:
            plt.scatter(points[:, 0], points[:, 1], color='skyblue', s=10, label='Nœuds du maillage')

        # Tracé des trajectoires de ce tour
        labels_done = {'Na+': False, 'Cl-': False, 'H20': False}
        for p in particles_data:
            if lap <= len(p['trajectoires']):
                traj = p['trajectoires'][lap-1]
        
                label = None
                if not labels_done[p['type']]:
                    label = p['type']
                    labels_done[p['type']] = True
        
                plt.plot(traj[:,0], traj[:,1],
                         color=p['couleur'],
                         linewidth=0.7,
                         alpha=0.7,
                         label=label)

        # Tracé de la bobine
        theta = np.linspace(0, 2*np.pi, 300)
        x_circ = x_coil + R_coil * np.cos(theta)
        y_circ = y_coil + R_coil * np.sin(theta)
        plt.plot(x_circ, y_circ, 'k', alpha=0.5, label="Bobine magnétique")

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Trajectoires du tour {lap}")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()