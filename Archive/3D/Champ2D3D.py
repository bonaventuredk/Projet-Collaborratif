import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# CONSTANTES
# ============================================================

mu0 = 4*np.pi*1e-7
I = 1.0
R_coil = 0.0012

N = 10
spacing = 0.001

z_centers = np.linspace(-spacing*(N-1)/2, spacing*(N-1)/2, N)

# ============================================================
# CHAMP D'UNE SPIRE
# ============================================================

def B_spire_3D(x, y, z, z0=0):

    rho = np.sqrt(x**2 + y**2)
    z_prime = z - z0

    if rho < 1e-12:
        return 0,0,(mu0*I*R_coil**2)/(2*(R_coil**2+z_prime**2)**(3/2))

    r1_sq = (R_coil-rho)**2 + z_prime**2
    r2_sq = (R_coil+rho)**2 + z_prime**2

    k_sq = 1 - r1_sq/r2_sq

    C = mu0*I/(2*np.pi*np.sqrt(r2_sq))

    K = ellipk(k_sq)
    E = ellipe(k_sq)

    F = (R_coil**2 + rho**2 + z_prime**2)/r1_sq

    B_rho = C*(z_prime/rho)*(F*E - K)

    Bz = C*(((R_coil**2 - rho**2 - z_prime**2)/r1_sq)*E + K)

    Bx = B_rho*(x/rho)
    By = B_rho*(y/rho)

    return Bx,By,Bz


# ============================================================
# CHAMP TOTAL
# ============================================================

def B_total(x,y,z):

    Bx=By=Bz=0

    for z0 in z_centers:

        bx,by,bz = B_spire_3D(x,y,z,z0)

        Bx+=bx
        By+=by
        Bz+=bz

    return Bx,By,Bz


# ============================================================
# CHAMP DANS LE PLAN x-z
# ============================================================

x = np.linspace(-0.1,0.1,80)
z = np.linspace(-0.1,0.1,80)

X,Z = np.meshgrid(x,z)

Bx = np.zeros_like(X)
Bz = np.zeros_like(Z)

for i in range(len(x)):
    for j in range(len(z)):

        bx,by,bz = B_total(X[j,i],0,Z[j,i])

        Bx[j,i]=bx
        Bz[j,i]=bz


plt.figure(figsize=(7,7))

plt.streamplot(X,Z,Bx,Bz,density=1.5)

# dessin des spires comme segments horizontaux
L = spacing  # longueur visuelle du segment

for z0 in z_centers:
    plt.plot([-L, L], [z0, z0], color='red', linewidth=3)

plt.title("Champ magnétique dans le plan x-z")

plt.xlabel("x (m)")
plt.ylabel("z (m)")


plt.xlim(-0.1,0.1)
plt.ylim(-0.1,0.1)

plt.show()


# ============================================================
# CHAMP 3D
# ============================================================

grid = np.linspace(-0.1,0.1,8)

X,Y,Z = np.meshgrid(grid,grid,grid)

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
Bz = np.zeros_like(Z)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):

            bx,by,bz = B_total(X[i,j,k],Y[i,j,k],Z[i,j,k])

            Bx[i,j,k]=bx
            By[i,j,k]=by
            Bz[i,j,k]=bz


fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111,projection='3d')

ax.quiver(X,Y,Z,Bx,By,Bz,length=0.03,normalize=True)


# dessin des spires
theta = np.linspace(0,2*np.pi,200)

for z0 in z_centers:

    x_coil = R_coil*np.cos(theta)
    y_coil = R_coil*np.sin(theta)
    z_coil = np.ones_like(theta)*z0

    ax.plot(x_coil,y_coil,z_coil,color='red',linewidth=2)


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim(-0.1,0.1)
ax.set_ylim(-0.1,0.1)
ax.set_zlim(-0.1,0.1)

ax.set_title("Champ magnétique 3D des bobines")

plt.show()