"""
Created on Fri Jan 30 12:14:11 2026

@author: bonaventure & audrey & fisk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.integrate import solve_ivp


D = 0.019
L = 0.19

# =========================
# PARAMÈTRES DES BOBINES
# =========================
# Constantes physiques et paramètres géométriques
R_coil = D/3          # Rayon de la spire (en mètres)
I = 1 # Courant traversant la spire (en ampères)   <---------------
N = 100            # Nombre total de spires
spacing = 0.001    # Espacement entre les spires (en mètres)
mu0 = 4 * np.pi * 1e-7   # Perméabilité magnétique du vide (H/m)
B0 = 200e-7  #

x_coil = 0.0
y_coil = D/2   
z_coil = -0.054

x_coils = np.array([0.0, 0.0])
z_coils = np.array([0.073, 0.073])

z0=z_coil 

def B_spire(x, z, z0=-0.0):
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

    global x_coil

    # Distance radiale au centre de la spire (projection dans le plan x)
    #rho = np.sqrt(np.square(x - x_coil) + np.square(y- y_coil)+ np.square(z - z_coil))
    #rho = np.sqrt(np.square(x))
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
    z_centers = np.linspace(-spacing*(N-1)/2 + z0, spacing*(N-1)/2 + z0, N)

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


Bx, Bz = N_B(x_coils, z_coils)
        
B = np.array([Bx,0.0,Bz]) / B0
print(B)


x = np.linspace(0, L, 80)
z = np.linspace(0, D, 80)
#x = np.linspace(-50*R_coil, 55*R_coil, 80)
#z = np.linspace(-50*R_coil +z0, 55*R_coil + z0, 80)
X, Z = np.meshgrid(x, z)

Bx = np.zeros_like(X)
By = np.zeros_like(X)
Bz = np.zeros_like(Z)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        bx, bz = B_N_spires(X[i,j] - x_coil, Z[i,j] - z_coil)
        Bx[i,j] += bx
        Bz[i,j] += bz

plt.figure(figsize=(7,7))
strm = plt.streamplot(X, Z, Bx, Bz, density=2.5, linewidth=0.7, arrowsize=1.2, color=np.sqrt(Bx**2 + Bz**2))

plt.colorbar(strm.lines, label="|B|")


#theta = np.linspace(0, 2*np.pi, 200)
#z_centers = np.linspace(-spacing*(N-1)/2 + z0, spacing*(N-1)/2 + z0, N)
#for i, zc in enumerate(z_centers):
#    if i == 0:
#        plt.plot(R_coil*np.cos(theta) + x_coils[0], np.zeros_like(theta)+zc, 'r', linewidth=3, label=f'Bobine ({N} spires)')
#    else:
#        plt.plot(R_coil*np.cos(theta) + x_coils[0], np.zeros_like(theta)+zc, 'r', linewidth=3)

plt.xlabel("x (m)")
plt.ylabel("z (m)")
#plt.title(f"Lignes de champ - {N} spires (espacement = {spacing} m)")
plt.title(f"Magnitude du champs magnétique dans le tube pour deux bobines situé à 0.0002 m du tube")
plt.legend()
#plt.axis()
plt.grid(alpha=0.3)
plt.show()
