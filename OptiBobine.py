import numpy as np 

from scipy.special import ellipe, ellipk

from SolveParticule5 import run_multi_lap


#### Paramlètres physiques et géométriques

L = 0.027              # Longueur totale du domaine (m)
D = 0.0036             # Hauteur totale du domaine (m)
lwall = D / 10         # Épaisseur caractéristique du mur latéral
Lwall = L / 8          # Longueur caractéristique du mur
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
z_coils = np.array([-0.052, 0.052])

#### Normalisation 

B0s = np.linspace(0,1e-4,N)

bobine = {"I" : 0,
          "Rayon" :0,
          "nb_spire":0,
          "spacing" :0,
          "x_coil" : 0,
          "y_coil" : 0,
          "z_coil" : 0,
          "B0" : 0
}


def parameters(I, Rayons, Nb_spires, Spaciing_spire, x_coils, y_coils, z_coils, B0s):
    return [I,Rayons, Nb_spires, Spaciing_spire, x_coils, y_coils, z_coils, B0s]

def optimize(parameters_list): 

    dt = 1e-3                     # Pas de temps (s)
    n1 = 0                        # Nombre de particules neutres (H20)
    n2 = 100                        # Nombre de particules positives (Na+)
    n3 = 100                        # Nombre de particules négatives (Cl-)
    total_laps = 1                 # Nombre maximum de tours à simuler
    number_of_runs = 10            #Nombre de simulations à lancer

    n_particles = {'Na+': n2, 'Cl-': n3, 'H20': n1}
    TOTAL_PARTICULES = n1 + n2 + n3

    bobine["I"] = parameters_list[0]
    bobine["Rayon"] = parameters_list[1]
    bobine["Nb_spire"] = parameters_list[2]
    bobine["spacing"] = parameters_list[3]
    bobine["x_coil"] = parameters_list[4]
    bobine["y_coil"] = parameters_list[5]
    bobine["z_coil"] = parameters_list[6]
    bobine["B0"] = parameters_list[7]

    
    for p in parameters_list: 
        #results = solve_particule(p)
        _, __, bilan_detail_par_tour = run_multi_lap(n_particles, total_laps, dt, bobine, verbose=False)

        Na_in_bot = 0.0
        Cl_in_top = 0.0

        for lap in range(number_of_runs):
            stats_detail = bilan_detail_par_tour[0]
            Na_in_bot += stats_detail["bas (de Na+)"]["Na+"]
            Cl_in_top += stats_detail["haut (de Cl-)"]["Cl-"]
        
        Na_in_bot /= number_of_runs
        Cl_in_top /= number_of_runs

    return objective(Na_in_bot,Cl_in_top, Na_tot=n2, Cl_tot=n3)

def objective(Na_in_bot, Cl_in_top, Na_tot, Cl_tot):
    return (Na_in_bot + Cl_in_top)/(Na_tot+Cl_tot)


def main():
    optimal_parameters = bobine
    max_score= 0.0
    print(f"Initial parameters: {optimal_parameters} with score {max_score:.4f}")

    #optimisateur naif
    """ 
    for I in Is: 
        for rayon in R_coils: 
            for nb_spire in Nb_spires: 
                for spacing in spacing_spires: 
                    for x_coil in x_coils: 
                        for y_coil in y_coils: 
                            for z_coil in z_coils: 
                                for B0 in B0s: 
                                    parameters_list = parameters(I, rayon, nb_spire, spacing, x_coil, y_coil, z_coil, B0)
                                    score = optimize(parameters_list)
                                    if score > max_score:
                                        max_score = score
                                        optimal_parameters = parameters_list
                                        print(f"New optimal parameters found with score {score:.4f}: {optimal_parameters}")
    """

    #Monte carlo
    number_of_runs = 10
    for _ in range(number_of_runs):
        I = np.random.uniform(0, Imax)
        rayon = np.random.uniform(0., D)
        nb_spires = np.random.randint(0, 1000)
        spacing = np.random.uniform(0, 0.01)
        x_coil = np.random.uniform(0, L - L/8)
        y_coil = np.random.uniform(-D/2, D/2)
        z_coil = np.random.choice(z_coils) 
        B0 = np.random.uniform(0,1e-4)

        parameters_list = parameters(I, rayon, nb_spires, spacing, x_coil, y_coil, z_coil, B0)
        print(f"Testing parameters: {parameters_list}")
        score = optimize(parameters_list)
        print(f"Tested parameters: {parameters_list} with score {score:.4f}")
        if score > max_score:
            max_score = score
            optimal_parameters = parameters_list
            print(f"New optimal parameters found with score {score:.4f}: {optimal_parameters}")

    return 0


main()