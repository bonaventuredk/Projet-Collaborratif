import numpy as np
from scipy.optimize import minimize



from SolveParticule5 import run_multi_lap, draw_domain



Ls = np.linspace(0.038, 0.19, 9, endpoint=True) # Longueur totale du domaine (m)
Ls[1] = 0.057
Ls[4] = 0.114
print(Ls)

#### Paramlètres physiques et géométriques

L = 0.27              # Longueur totale du domaine (m)
D = 0.019             # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = L / 8          # Longueur caractéristique du mur
Lwall = D
eta = D / 10           # Paramètre géométrique pour définir Rtip
Rtip = eta / 2         # Rayon de l'extrémité (tip)
delta = 6 * Rtip       # Largeur de la zone d'influence autour du tip
xmur = L - Lwall       # Position du mur magnétique (début des pointes)
mu0 = 4 * np.pi * 1e-7 

N = 1000
Imax = 10 

Is = np.linspace(0, Imax, N)
R_coils = np.linspace(0., D, N) # plus 
Nb_spires = np.linspace(0, 1000, N)
spacing_spires = np.linspace(0, 0.01, N) # Plus petit que la te

#### Positionnement des bobines 

x_coils = np.linspace(0, L - L/8, N)
y_coils = np.linspace(-D/2, D/2, N)
z_coils = np.array([-0.054, 0.073]) # Position des bobines en z (en bas et en haut du domaine)

x_coil = 0.0
y_coil = D/2

#### Normalisation 

B0s = np.linspace(0,1e-4,N)

bobine = {"L": 0,
          "I" : 0,
          "Rayon" :0,
          "nb_spire":0,
          "spacing" :0,
          "x_coil" : 0,
          "y_coil" : 0,
          "z_coil" : 0,
          "B0" : 0
}

def parameters(L,I, Rayons, Nb_spires, Spaciing_spire, x_coils, y_coils, z_coils, B0s):
    return [L,I,Rayons, Nb_spires, Spaciing_spire, x_coils, y_coils, z_coils, B0s]

def optimize(parameters): 

    print("x[0] :", parameters[0])

    dt = 1e-3                     # Pas de temps (s)
    n1 = 0                        # Nombre de particules neutres (H20)
    n2 = 100                        # Nombre de particules positives (Na+)
    n3 = 100                        # Nombre de particules négatives (Cl-)
    total_laps = 1                 # Nombre maximum de tours à simuler
    number_of_runs = 1            #Nombre de simulations à lancer

    n_particles = {'Na+': n2, 'Cl-': n3, 'H20': n1}
    TOTAL_PARTICULES = n1 + n2 + n3

    Rayons = D/3
    Nb_spires = 100
    Spaciing_spire = 0.001
    L = 0.076
    x_coil = 0.0
    y_coil = D/2
    z_coil = -0.054
    B0 = 200e-7



    bobine["L"] = L   
    bobine["I"] = parameters[0]
    bobine["Rayon"] = parameters[1]
    bobine["Nb_spire"] = N
    bobine["spacing"] = parameters[2]
    bobine["x_coil"] = x_coil
    bobine["y_coil"] = y_coil
    bobine["z_coil"] = z_coil
    bobine["B0"] = parameters[3]

    
  
    #results = solve_particule(p)
    print("bobine :", bobine)
    print("parameters :", parameters)
    _, __, bilan_detail_par_tour = run_multi_lap(n_particles, total_laps, dt, bobine, verbose=False)

    Na_in_bot = 0.0
    Cl_in_top = 0.0

    for lap in range(number_of_runs):
        stats_detail = bilan_detail_par_tour[0]
        Na_in_bot += stats_detail["bas (de Na+)"]["Na+"]
        Cl_in_top += stats_detail["haut (de Cl-)"]["Cl-"]
        
    Na_in_bot /= number_of_runs
    Cl_in_top /= number_of_runs

    print("score :", objective(Na_in_bot,Cl_in_top, Na_tot=n2, Cl_tot=n3))

    return objective(Na_in_bot,Cl_in_top, Na_tot=n2, Cl_tot=n3)

def objective(Na_in_bot, Cl_in_top, Na_tot, Cl_tot):
    return (Na_in_bot + Cl_in_top)/(Na_tot+Cl_tot)

#cons = ({'A':'N', 'B':'M'},{'A':'N', 'B':'M'})

#cons = ({'type':'ineq', 'fun': lambda x : x - 10 })
cons = None

bnds = [(0,10),(0,D),(1e-5, 1e-2),(1e-7,1e-4)]

mean_B0 = (1e-7 + 1e-4)/2
var_B0 = (1e-4 - 1e-7)

mean_spacing = (0.01 + 0.0001)/2
var_spacing = (0.01 - 0.0001)


x0 = np.zeros((4))
x0[0] = 0.5
x0[1] = (D/3)/D
x0[2] = (0.001)/var_spacing
x0[3] = (200e-7)/var_B0

x0[0] = 1
x0[1] = (D/3)
x0[2] = (0.001)
x0[3] = (200e-7)

print("x[0] :", x0[0])


print("Before Solving")
result = minimize(optimize, x0, bounds = bnds, constraints= cons, method = 'BFGS')
print("results :", result)