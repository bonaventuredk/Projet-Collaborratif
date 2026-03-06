"""
Created on Fri Jan 30 12:14:11 2026

@author: bonaventure & audrey & thomas
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.interpolate import CloughTocher2DInterpolator

# =========================================
# CHARGEMENT DES DONNÉES FREEFEM
# =========================================
def load_freefem_data(nodes_file, ux_file, uy_file):
    """
    Charge les données issues d’un calcul FreeFem et construit des interpolateurs
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
    ux_interp : CloughTocher2DInterpolator
        Interpolateur 2D pour la composante ux.
    uy_interp : CloughTocher2DInterpolator
        Interpolateur 2D pour la composante uy.
    points : ndarray
        Tableau des coordonnées des nœuds.

    Notes
    -----
    - Les fichiers doivent contenir un nombre identique de lignes.
    - En cas d’erreur (fichier manquant, format incorrect, etc.), la fonction
      renvoie trois valeurs None.
    """
    
    # Message d'information pour indiquer le début du chargement
    print("Chargement des données FreeFem...")

    try:
        # Chargement des coordonnées des nœuds (tableau Nx2)
        points = np.loadtxt(nodes_file)

        # Chargement de la composante horizontale de la vitesse
        ux_data = np.loadtxt(ux_file)

        # Chargement de la composante verticale de la vitesse
        uy_data = np.loadtxt(uy_file)

        # Vérification que le nombre de nœuds correspond au nombre de valeurs de vitesse
        if len(points) != len(ux_data):
            raise ValueError("Nombre de nœuds différent du nombre de vitesses.")

        # Message indiquant la création des interpolateurs
        print("Création des interpolateurs...")

        # Construction de l’interpolateur 2D pour ux
        ux_interp = CloughTocher2DInterpolator(points, ux_data)

        # Construction de l’interpolateur 2D pour uy
        uy_interp = CloughTocher2DInterpolator(points, uy_data)

        #print("Interpolation is useless :", ux_data == ux_interp)
        #print("Interpolation is useless :", uy_data == uy_interp)

        # Message de confirmation
        print("Chargement terminé ")

        # Retourne les deux interpolateurs ainsi que les points
        return ux_interp, uy_interp, points

    except Exception as e:
        # Affiche l’erreur rencontrée (lecture fichier, format, etc.)
        print("Erreur :", e)

        # Retourne des valeurs nulles en cas d’échec
        return None, None, None

# =========================
# PARAMÈTRES DU DOMAINE
# =========================
L = 0.027              # Longueur totale du domaine (m)
D = 0.0036             # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = L / 8          # Longueur caractéristique du mur
eta = D / 10           # Paramètre géométrique utilisé pour définir Rtip
Rtip = eta / 2         # Rayon de l'extrémité (tip)
delta = 6 * Rtip       # Largeur d'une zone d'influence autour du tip
xmur = L - Lwall       # Position du mur magnétique
x_coil = L / 10         # Longitude de la bobine
z_coil = 0.0           # Hauteur de la bobine

x_coils = np.array([L/10, L/5,])
z_coils = np.array([0.0, 0.0])


####### Creations des objets murs 


theta = np.linspace(3*np.pi/2, np.pi/2, 300)
x_tip = delta*np.cos(theta) + xmur
x_tip = np.expand_dims(x_tip, axis = 1)
y_tip_bottom = Rtip*np.sin(theta) + lwall + Rtip
y_tip_bottom = np.expand_dims(y_tip_bottom, axis = 1)
y_tip_top = Rtip*np.sin(theta) + D - lwall - eta + Rtip
y_tip_top = np.expand_dims(y_tip_top, axis = 1)

x_wall = np.expand_dims(np.linspace(L-L/8, L, 100), axis=1)

bot_bot_wall = np.concatenate((x_wall, (D/10)*np.ones_like(x_wall)),axis = 1)
mid_bot_wall = np.concatenate((x_tip, y_tip_bottom),axis = 1)
top_bot_wall = np.concatenate((x_wall, (D/10+ lwall)*np.ones_like(x_wall)),axis = 1)
bot_wall = np.concatenate((bot_bot_wall, mid_bot_wall, top_bot_wall), axis = 0)

bot_top_wall = np.concatenate((x_wall, (D - 2*D/10)*np.ones_like(x_wall)),axis = 1)
mid_top_wall = np.concatenate((x_tip, y_tip_top),axis = 1)
top_top_wall = np.concatenate((x_wall, (D- 2*D/10+ lwall)*np.ones_like(x_wall)),axis = 1)
top_wall = np.concatenate((bot_top_wall, mid_top_wall, top_top_wall), axis = 0)



# =========================
# PARAMÈTRES DES BOBINES
# =========================
# Constantes physiques et paramètres géométriques
R_coil = D/3          # Rayon de la spire (en mètres)
I = 1              # Courant traversant la spire (en ampères)
N = 100            # Nombre total de spires
spacing = 0.001    # Espacement entre les spires (en mètres)
mu0 = 4 * np.pi * 1e-7   # Perméabilité magnétique du vide (H/m)


def B_spire(x, z, z0=0.0):
    """
    Calcule le champ magnétique (Bx, Bz) produit par une spire circulaire
    de rayon R, parcourue par un courant I, au point (x, z).

    Paramètres
    ----------
    x : float
        Coordonnée horizontale du point d'observation.
    z : float
        Coordonnée verticale du point d'observation.
    z0 : float, optionnel
        Position verticale de la spire (par défaut 0).

    Retour
    ------
    Bx : float
        Composante horizontale du champ magnétique.
    Bz : float
        Composante verticale du champ magnétique.

    Notes
    -----
    - Le calcul utilise les formules classiques basées sur les intégrales
      elliptiques complètes K(k) et E(k).
    - Un traitement particulier est appliqué lorsque x = 0 pour éviter
      une division par zéro.
    """

    # Distance radiale au centre de la spire (projection dans le plan x)
    rho = np.abs(x)

    # Décalage vertical par rapport au plan de la spire
    z_prime = z - z0

    # Cas particulier : point situé exactement sur l’axe de la spire
    if rho < 1e-12:
        Bx = 0.0  # Symétrie axiale → composante horizontale nulle

        # Formule analytique du champ sur l’axe d’une spire
        Bz = (mu0 * I * R_coil**2) / (2 * (R_coil**2 + z_prime**2)**(3/2))
        return Bx, Bz

    # Distances au carré pour les formules elliptiques
    r1_sq = (R_coil - rho)**2 + z_prime**2
    r2_sq = (R_coil + rho)**2 + z_prime**2

    # Paramètre elliptique k²
    k_sq = 1 - r1_sq / r2_sq

    # Facteur commun dans les formules
    C = mu0 * I / (2 * np.pi * np.sqrt(r2_sq))

    # Intégrales elliptiques complètes de première et seconde espèce
    K = ellipk(k_sq)
    E = ellipe(k_sq)

    # Facteur géométrique utilisé dans les expressions
    F = (R_coil**2 + rho**2 + z_prime**2) / r1_sq

    # Composante radiale du champ magnétique
    B_rho = C * (z_prime / rho) * (F * E - K)

    # Composante verticale du champ magnétique
    B_z = C * (((R_coil**2 - rho**2 - z_prime**2) / r1_sq) * E + K)

    # Conversion de B_rho en composante Bx selon le signe de x
    Bx = B_rho * np.sign(x) if x != 0 else 0.0

    return Bx, B_z



def B_N_spires(x, z):
    """
    Calcule le champ magnétique total (Bx, Bz) produit par N spires
    identiques, régulièrement espacées verticalement.

    Paramètres
    ----------
    x : float
        Coordonnée horizontale du point d'observation.
    z : float
        Coordonnée verticale du point d'observation.

    Retour
    ------
    Bx_total : float
        Composante horizontale totale du champ magnétique.
    Bz_total : float
        Composante verticale totale du champ magnétique.

    Notes
    -----
    - Les spires sont centrées autour de z = 0.
    - Chaque spire est séparée de la suivante par `spacing`.
    """

    # Initialisation des composantes du champ total
    Bx_total, Bz_total = 0.0, 0.0

    # Positions verticales des centres des N spires
    z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)

    # Somme des contributions de chaque spire
    for z_center in z_centers:
        bx, bz = B_spire(x, z, z_center)
        Bx_total += bx
        Bz_total += bz

    return Bx_total, Bz_total


def N_B(x_coils , z_coils):
    """
    Calcule le champ magnétique total (Bx, Bz) produit par N bobines de n spires
    placées sur les points de coordonnées x_coils et z_coils.

    Paramètres
    ----------
    x_coils : float
        Array contenant les coordonnées horizontales des points d'observations.
    z_coils : float
        Array contenant les coordonnées verticales des points d'observations.

    Retour
    ------
    Bx_total : float
        Composante horizontale totale du champ magnétique.
    Bz_total : float
        Composante verticale totale du champ magnétique.

    Notes
    -----
    - Les spires sont centrées autour de z = 0.
    - Chaque spire est séparée de la suivante par `spacing`.
    """
    Bx_total, Bz_total = 0.0 , 0.0 

    for x, z in (x_coils, z_coils) :
        bx, bz = B_N_spires(x,z)
        Bx_total += bx
        Bz_total += bz

    return Bx_total, Bz_total

# =========================
# FONCTION D'AMORTISSEMENT 
# =========================

def damping_y(y, Uy, D, charge_sign, gamma_max=1000.0, delta_zone=4e-4):
    """
    Amortissement sélectif selon :
    - mur du bas  -> agit seulement sur charges positives
    - mur du haut -> agit seulement sur charges négatives
    
    Notes:
        Astuces pour éviter que les particules soient attiré par le mauvais bord.
    """

    damping = 0.0

    # =========================
    # MUR DU BAS (y = 0)
    # =========================
    if y < delta_zone:
        # agit seulement si charge positive
        if charge_sign > 0:
            damping = gamma_max * (delta_zone - y) / delta_zone
        else:
            return 0.0  # aucune action

    # =========================
    # MUR DU HAUT (y = D)
    # =========================
    elif y > D - delta_zone:
        # agit seulement si charge négative
        if charge_sign < 0:
            damping = gamma_max * (y - (D - delta_zone)) / delta_zone
        else:
            return 0.0  # aucune action

    return -damping * Uy

def repel_from_wall(X, U, wall, d_min=0.001, strength=0.1):

    X = X[:2]
    U = U[:2]
    
    # compute distances to all wall points
    diff = wall - X
    dist = np.linalg.norm(diff, axis=1)
    
    # find closest point
    idx = np.argmin(dist)
    d = dist[idx]
    
    if d < d_min:
        normal = X - wall[idx]
        normal = normal / np.linalg.norm(normal)

        U = U - 2*strength*np.dot(U, normal)*normal  # reflection
        
    return U[:2]

# =========================
# SIMULATION
# =========================
Nt = 1000
dt = 1e-3
B0 = 200e-7

def simulate_particle(X0, U0, charge_sign=1):
    X = np.zeros((Nt,3))
    U = np.zeros((Nt,3))
    X[0] = X0
    U[0] = U0

    for i in range(Nt-1):
        # Champ magnétique
        # ici la position de la bobine influence ..., bien regarder 
        global x_coils, z_coils
        
        Bx, Bz = 0.0 , 0.0
        for x, z in x_coils, z_coils :
            bx, bz = B_N_spires(X[i,0] - x, X[i,2] - z)
            Bx += bx 
            Bz += bz
        B = np.array([Bx, 0.0, Bz]) / B0

        # ODE Lorentz
        def f(u):
            return charge_sign * np.cross(u, B)

        # Méthode semi-implicite
        U_int = U[i] + dt*f(U[i])
        U_new = U[i] + dt/2*(f(U[i]) + f(U_int))

        # Contourner les murs au fond 
        U_new[:2] = repel_from_wall(X[i], U_new, bot_wall, d_min = 1e-3)
        U_new[:2] = repel_from_wall(X[i], U_new, top_wall, d_min = 1e-3)

        # Amortissement vertical
        U_new[1] += dt * damping_y(X[i,1], U_new[1], D, charge_sign)


        # Mise à jour
        X[i+1] = X[i] + dt*U_new
        U[i+1] = U_new

    return X, U

# =========================
# PARTICULES MULTIPLES (choix utilisateur)
# =========================

# Nombre de particules
n1 = 5   # charge nulle
n2 = 5   # charge positive (les Na+)
n3 = 5  # charge négative (les Cl-)
TOTALPARTICULES = n1 + n2 + n3

def generate_particles(n, charge_sign):
    X_list = []
    U_list = []

    b = 100
    y0 = np.logspace(0, 1, num=n, base =b) 


    if charge_sign <0 : 
        
        y0 += charge_sign*np.ones((n))
        
        y0 = (D/(b-1))*y0


    elif charge_sign > 0 : 

        y0 -= np.ones((n))

        y0 = (D/(b-1))*y0

        y0 *= -charge_sign

        y0 += D*np.ones((n))
    
    else :
        y0 = np.random.uniform(0,D, size=(n))


    
    for i in range(n):
        # Position aléatoire sur le segment (0,0) -> (0,D)
        #y0 = np.random.uniform(0, D)
        X0 = np.array([0.0, y0[i], 0.0])
        
        # Vitesse initiale (modifiable si besoin)
        U0 = np.array([1.0, 0.0, 0.0])
        
        X, U = simulate_particle(X0, U0, charge_sign=charge_sign)
        X_list.append(X)
        U_list.append(U)
        
    return X_list, U_list

# Génération des particules
X_green_list, _ = generate_particles(n1, 0)
X_blue_list, _  = generate_particles(n2, 1)
X_red_list, _   = generate_particles(n3, -1)





# =========================
# TRACÉ DU DOMAINE
# =========================
def draw_domain():
    plt.plot([0, L], [0, 0], 'k')
    plt.plot([0, L], [D, D], 'k')
    plt.plot([0, 0], [0, D], 'k')
    plt.plot([L, L], [0, lwall], 'k')
    plt.plot([L, L], [lwall+eta, D-lwall-eta], 'k')
    plt.plot([L, L], [D-lwall, D], 'k')
    plt.plot([xmur, L], [lwall, lwall], 'k')
    plt.plot([xmur, L], [D-lwall, D-lwall], 'k')
    plt.plot([xmur, L], [lwall+eta, lwall+eta], 'k')
    plt.plot([xmur, L], [D-lwall-eta, D -lwall-eta], 'k')
    theta = np.linspace(3*np.pi/2, np.pi/2, 300)
    x_tip = delta*np.cos(theta) + xmur
    y_tip_bottom = Rtip*np.sin(theta) + lwall + Rtip
    y_tip_top = Rtip*np.sin(theta) + D - lwall - eta + Rtip
    plt.plot(x_tip, y_tip_bottom, 'k')
    plt.plot(x_tip, y_tip_top, 'k')
    plt.plot([xmur-delta, xmur-delta], [0, lwall + eta/2], 'r--', linewidth=1) #pour les bleus
    plt.plot([xmur-delta, xmur-delta], [lwall + eta/2, D - lwall - eta/2], 'g-', linewidth=1) # pour le milieu
    plt.plot([xmur-delta, xmur-delta], [D - lwall-eta/2, D], 'b--', linewidth=1) # pour les rouges

# =========================
# BILAN DES PARTICULES
# =========================


# Limites des segments
y_bleu_max   = lwall + eta/2
y_milieu_min = lwall + eta/2
y_milieu_max = D - lwall - eta/2
y_rouge_min  = D - lwall - eta/2

# Initialisation des compteurs par zone
bilan = {
    "bas (de Na+)"   : {"Na+": 0, "Cl-": 0, "H20": 0},
    "milieu": {"Na+": 0, "Cl-": 0, "H20": 0},
    "haut (de Cl-)"  : {"Na+": 0, "Cl-": 0, "H20": 0}
}

# Fonction pour déterminer la zone d'une particule
def zone_particule_croisee(particule):
    y = particule[-1,1]  # dernière coordonnée y
    if y <= y_bleu_max:
        return "bas (de Na+)"
    elif y >= y_rouge_min:
        return "haut (de Cl-)"
    elif y_milieu_min <= y <= y_milieu_max:
        return "milieu"
    else:
        return "inconnu"

# Fonction pour identifier la couleur
def couleur_particule(X):
    # Vérifie à quelle liste elle appartient
    if any(np.array_equal(X, xb) for xb in X_blue_list):
        return "Na+"
    elif any(np.array_equal(X, xr) for xr in X_red_list):
        return "Cl-"
    else:
        return "H20"

# Parcours de toutes les particules
toutes_particules = X_blue_list + X_red_list + X_green_list

for X in toutes_particules:
    zone = zone_particule_croisee(X)
    couleur = couleur_particule(X)
    if zone in bilan:
        bilan[zone][couleur] += 1

# Affichage du bilan
print("===== BILAN DES PARTICULES =====")


print(f"Nombre totales de particules: {TOTALPARTICULES}")

for zone in ["bas (de Na+)", "milieu", "haut (de Cl-)"]:
    total_zone = sum(bilan[zone].values())
    print(f"\nZone {zone} (total {total_zone}) :")
    for couleur in ["Na+", "Cl-", "H20"]:
        nb = bilan[zone][couleur]
        pct = nb / total_zone * 100 if total_zone>0 else 0
        print(f"  {couleur} : {nb} ({pct:.1f}%)")

# =========================
# AFFICHAGE
# =========================
ux_interp, uy_interp, points = load_freefem_data('nodes.txt', 'ux.txt', 'uy.txt')
plt.figure(figsize=(10,5))
draw_domain()
# Nœuds du maillage
if points is not None:
    plt.scatter(points[:,0], points[:,1], color='skyblue', s=10, label='Nœuds du maillage')

# Nombre de points affichés
n_display = 28

# q < 0 (rouge)
for i, X in enumerate(X_red_list):
    if i == 0:
        plt.plot(X[:n_display,0], X[:n_display,1], 'r', linewidth=0.7, label='Trajectoires de Cl-')
    else:
        plt.plot(X[:n_display,0], X[:n_display,1], 'r', linewidth=0.7)

# q > 0 (bleu)
for i, X in enumerate(X_blue_list):
    if i == 0:
        plt.plot(X[:n_display,0], X[:n_display,1], 'b', linewidth=0.7, label='Trajectoires de Na+')
    else:
        plt.plot(X[:n_display,0], X[:n_display,1], 'b', linewidth=0.7)

# q = 0 (vert)
for i, X in enumerate(X_green_list):
    if i == 0:
        plt.plot(X[:n_display,0], X[:n_display,1], 'g', linewidth=0.7, label='Trajectoires q=0 (H20)')
    else:
        plt.plot(X[:n_display,0], X[:n_display,1], 'g', linewidth=0.7)

# Bobine (optionnel)
theta = np.linspace(0, 2*np.pi, 300)

for x_coil in x_coils :
    x_circ = x_coil + R_coil*np.cos(theta)
    y_circ = D/2 + R_coil*np.sin(theta)
    plt.plot(x_circ, y_circ, 'r', label='Bobine')

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectoires des particules dans le tube XY")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()