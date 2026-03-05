import numpy as np

D = 0.0036
charge_sign = -1 
n = 100
b = 100

y0 = np.logspace(0, 1, num=n, base =b) 


if charge_sign <0 : 
    y0 += charge_sign*np.ones((n))

    print("y0 :", y0)

    y0 = (D/(b-1))*y0

    print("charge_sign :", charge_sign)
    print("y0 :", y0)

else : 

    y0 -= np.ones((n))

    y0 = (D/(b-1))*y0

    print("charge_sign :", charge_sign)
    print("y0 :", y0)

    y0 *= -charge_sign

    y0 += D*np.ones((n))
    print("y0 :", y0)


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


# =========================
# PARAMÈTRES DU DOMAINE 2D
# =========================
L = 0.027              # Longueur totale du domaine (m)
D = 0.0036             # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = L / 8          # Longueur caractéristique du mur
eta = D / 10           # Paramètre géométrique utilisé pour définir Rtip
Rtip = eta / 2         # Rayon de l'extrémité (tip)
delta = 6 * Rtip       # Largeur d'une zone d'influence autour du tip
xmur = L - Lwall       # Position du mur magnétique
x_coil = L / 100         # Longitude de la bobine
z_coil = 0.0           # Hauteur de la bobine


pos_wall_bot = D/10 # Position du mur en bas 
pos_wall_top = D- 2*D/10
#Uy est la vitesse dans la direction y du fluide 

##### Pour le mur du bas
def detect_bot_wall(x,y, pos_wall = pos_wall_bot) :
    if x >= L - L/8 - delta :
        if np.abs(pos_wall - y) < delta :#La particule 
            damping = 9
            Uy 
            return True #Uy < 0 
        elif np.abs(y - (pos_wall + lwall)) < delta : 
            return True #Uy > 0
        if True :#point est près de l'ellipse  
            if y >= (pos_wall + lwall/2) :
                return True #Uy > 0
            if y <= (pos_wall + lwall/2) :
                return True #Uy < 0
        return False  


    return False

def detect_top_wall(x,y, pos_wall = pos_wall_top) :
    if x >= L - L/8 - delta :
        if np.abs(pos_wall - y) < delta :
            return True #Uy < 0 
        elif np.abs(y - (pos_wall + lwall)) < delta : 
            return True #Uy > 0
        if True :#point est près de l'ellipse  
            if y >= (pos_wall + lwall/2) :
                return True #Uy > 0
            if y <= (pos_wall + lwall/2) :
                return True #Uy < 0
        return False  


    return False  


theta = np.linspace(3*np.pi/2, np.pi/2, 300)
x_tip = delta*np.cos(theta) + xmur
y_tip_bottom = Rtip*np.sin(theta) + lwall + Rtip
y_tip_top = Rtip*np.sin(theta) + D - lwall - eta + Rtip
bot_wall = np.zeros((500,2))

x_wall = np.expand_dims(np.linspace(L-L/8, L, 100), axis=1)

bot_bot_wall = np.concatenate((x_wall, (D/10)*np.ones_like(x_wall)),axis = 1)
top_bot_wall = np.concatenate((x_wall, (D/10+ lwall)*np.ones_like(x_wall)),axis = 1)

x_tip = np.expand_dims(x_tip, axis = 1)
y_tip_bottom = np.expand_dims(y_tip_bottom, axis = 1)
mid_bot_wall = np.concatenate((x_tip, y_tip_bottom),axis = 1)

bot_wall = np.concatenate((bot_bot_wall, mid_bot_wall, top_bot_wall), axis = 0)

print("np.shape(bot_wall) :", np.shape(bot_wall))

bot_top_wall = np.concatenate((x_wall, (D - 2*D/10)*np.ones_like(x_wall)),axis = 1)
top_top_wall = np.concatenate((x_wall, (D- 2*D/10+ lwall)*np.ones_like(x_wall)),axis = 1)

y_tip_top = np.expand_dims(y_tip_top, axis = 1)
mid_top_wall = np.concatenate((x_tip, y_tip_top),axis = 1)

top_wall = np.concatenate((bot_top_wall, mid_top_wall, top_top_wall), axis = 0)

print("np.shape(top_wall) :", np.shape(top_wall))


import numpy as np

def repel_from_wall(x, y, Ux, Uy, wall, d_min=0.02, strength=0.01):

    particle = np.array([x,y])
    
    # compute distances to all wall points
    diff = wall - particle
    dist = np.linalg.norm(diff, axis=1)
    
    # find closest point
    idx = np.argmin(dist)
    d = dist[idx]

    U = np.array([Ux,Uy])
    
    if d < d_min:
        normal = particle - wall[idx]
        normal = normal / np.linalg.norm(normal)

        U = np.array([Ux, Uy])
        U = U - 2*np.dot(U, normal)*normal  # reflection
        
    return U


def point_segment_distance(p, a, b):

    ap = p - a
    ab = b - a

    t = np.dot(ap, ab) / np.dot(ab, ab)## = Norm(ap)*cos(PAB)
    t = np.clip(t, 0, 1) ##

    closest = a + t * ab ## trouve le point le plus proche de P appartenant au segment AB
    dist = np.linalg.norm(p - closest)

    return dist,closest

def point_segment_distance2(p, a, b):

    ap = p - a
    ab = b - a

    t = np.dot(ap, ab) / np.dot(ab, ab) ## = Norm(ap)*cos(PAB)
    t = np.clip(t, 0, 1) ##


    dist = ap*np.sqrt(1 - (t/np.linalg.norm(ap))**2)

    return np.linalg.norm(dist)


p = np.array([1, 4])
a= np.array([0,1])
b=np.array([0,3])


print("dist 1 : ", point_segment_distance(p,a,b))

print("dist 1 : ", point_segment_distance2(p,a,b))


