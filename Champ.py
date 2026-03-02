"""
Created on Fri Jan 30 12:14:11 2026

@author: bonaventure & audrey & fisk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.integrate import solve_ivp

# Paramètres physiques
R = 0.002
I = 50.0
mu0 = 4 * np.pi * 1e-7

N = 10
spacing = 0.002


def B_spire(x=0, z=0,R=0.005, I=50, z0=0, ):
    rho = np.abs(x)
    z_prime = z - z0
    if rho < 1e-12:
        Bx = 0.0
        Bz = (mu0 * I * R**2) / (2 * (R**2 + z_prime**2)**(3/2))
        return Bx, Bz
    r1_sq = (R - rho)**2 + z_prime**2
    r2_sq = (R + rho)**2 + z_prime**2
    k_sq = 1 - r1_sq / r2_sq
    C = mu0 * I / (2 * np.pi * np.sqrt(r2_sq))
    K = ellipk(k_sq)
    E = ellipe(k_sq)
    F = (R**2 + rho**2 + z_prime**2) / r1_sq
    B_rho = C * (z_prime / rho) * (F * E - K)
    B_z = C * (((R**2 - rho**2 - z_prime**2) / r1_sq) * E + K)
    Bx = B_rho * np.sign(x) if x != 0 else 0.0
    return Bx, B_z

def B_N_spires(x, z):
    Bx_total, Bz_total = 0.0, 0.0
    z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)
    for z_center in z_centers:
        bx, bz = B_spire(x, z,R,I, z_center)
        Bx_total += bx
        Bz_total += bz
    return Bx_total, Bz_total

x = np.linspace(-50*R, 55*R, 80)
z = np.linspace(-50*R, 55*R, 80)
X, Z = np.meshgrid(x, z)

Bx = np.zeros_like(X)
By = np.zeros_like(X)
Bz = np.zeros_like(Z)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Bx[i,j], Bz[i,j] = B_N_spires(X[i,j], Z[i,j])

plt.figure(figsize=(7,7))
plt.streamplot(X, Z, Bx, Bz, density=2.5, linewidth=0.7, arrowsize=1.2, color='darkblue')

theta = np.linspace(0, 2*np.pi, 200)
z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)
for i, zc in enumerate(z_centers):
    if i == 0:
        plt.plot(R*np.cos(theta), np.zeros_like(theta)+zc, 'r', linewidth=3, label=f'Bobine ({N} spires)')
    else:
        plt.plot(R*np.cos(theta), np.zeros_like(theta)+zc, 'r', linewidth=3)

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title(f"Lignes de champ - {N} spires (espacement = {spacing} m)")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.3)
plt.show()
