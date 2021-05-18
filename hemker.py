###############################################################################
#
# hemker.py
#
# a python script for modeling steady-state drawdown in response to pumping
# in a multi-aquifer system based on Hemker (1984) analytical solution
#
# by Walt McNab
#
###############################################################################
 
import numpy as np
import scipy.special
import scipy.spatial.distance
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
################################
#
# hydraulic properties classes
#
################################
 
class HydraulicUnits:
    def __init__(self, K, b):
        self.K = K           # hydraulic conductivity (horizontal or vertical)
        self.b = b           # vertical thickness
 
class Aquifers(HydraulicUnits):
    def __init__(self, K, b, z):
        HydraulicUnits.__init__(self, K, b)
        self.T = self.K * self.b                # transmissivity
        self.z = z                              # depth midpoint of aquifer
        self.L = np.zeros(len(K))                  # aquifer leakance
    def Leakance(self, w):
        self.L = np.sqrt(1.0/w)
 
class Aquitards(HydraulicUnits):
    def __init__(self, K, b):
        HydraulicUnits.__init__(self, K, b)
        self.c = b/K                            # vertical hydraulic resistance
 
class Well:
    def __init__(self, x, y, Q_tot, screened_list, aquifer):
        self.x = x                                  # well location
        self.y = y
        self.Q_tot = Q_tot                          # total injection rate
        self.screened_list = np.array(screened_list)   # screened flag
        self.Q = self.DistributeQ(aquifer)
    def DistributeQ(self,aquifer):
        # distribute Q among aquifers, based on aquifer transmissivity
        T_tot = (aquifer.T * self.screened_list).sum()
        Q = self.Q_tot * self.screened_list * aquifer.T/T_tot
        return Q
    def HeadChange(self, x, y, k, num_aqf, aquifers, V, Z):
        # head change in layer k resulting from pumping this well
        r = np.pdist([[self.x, self.y], [x, y]])[0]
        sum_term = np.zeros(num_aqf, float)
        for j in np.xrange(num_aqf):
            sum_term[j] = (V[k, :] * Z[j, :] * scipy.special.k0(r/aquifers.L)).sum()
        return (self.Q * sum_term/(2.0*np.pi*aquifers.T)).sum()
 
##################################
#
# Hemker model functions
#
##################################
 
def HemkerMatrix(aquifers, aquitards):
    # construct the Hemker model matrix
    a = 1.0/(aquifers.T * aquitards.c[:-1])
    b = 1.0/(aquifers.T * aquitards.c[1:])
    A = np.diag(a+b) + np.diag(-b[:-1], 1) + np.diag(-a[1:], -1)
    return A
 
def Eigen(A):
    # return eigenvalues and eigenvectors of matrix A and its transpose
    w, V = np.linalg.eig(A)
    Z = np.transpose(np.linalg.inv(V))
    return w, V, Z
 
##################################
#
# support functions
#
##################################
 
def GetLayers(b_aqf, b_aqt):
    # for user-specified aquifer-aquitard stratigraphy, process geometry
    section = b_aqf.sum() + b_aqt.sum()
    z_aqf = section - (b_aqt[:-1].cumsum() + b_aqf.cumsum() - 0.5*b_aqf)
    t_aqf = z_aqf + 0.5*b_aqf               # aquifer top elevation
    t_aqt = t_aqf + b_aqt[:-1]              # aquitard top elevation
    mixed_top = -np.concatenate((t_aqt, t_aqf))
    tops = -np.sort(mixed_top)
    tops = np.append(tops, b_aqt[-1])
    return z_aqf, tops
 
def Strat(section, num_contacts, K_aqf0, K_aqt0, K_inj, b_inj):
 
    # create strata
    rand = np.random.random(num_contacts) * section
    contact = np.sort(rand)
    tops = []                   # top of unit; used later for plotting
    K_aqf = []
    b_aqf = []
    z = []
    K_aqt = []
    b_aqt = []
    for i in np.xrange(len(contact)):          # stepping downward with depth
        if not i:
            # top layer --> aquitard, by definition
            K_aqt.append(K_aqt0)
            b_aqt.append(contact[i])
            tops.append(section)
        else:
            if (i % 2 == 0):
                # even: aquitard
                K_aqt.append(K_aqt0)
                b_aqt.append(contact[i] - contact[i-1])
            else:
                # odd: aquifer
                K_aqf.append(K_aqf0)
                b_aqf.append(contact[i] - contact[i-1])
                z.append(section - (0.5*contact[i-1] + 0.5*contact[i]))
            tops.append(section - contact[i-1])
 
    # append last zone in section (aquitard, by definition)
    K_aqt.append(K_aqt0)
    b_aqt.append(section - contact[-1])
    tops.append(section - contact[-1])
 
    # append injection zone, plus ficticious underlying aquitard
    K_aqf.append(K_inj)
    b_aqf.append(b_inj)
    z.append(-0.5*b_inj)
    tops.append(0.)
    K_aqt.append(K_aqt0)
    b_aqt.append(1000.)
    tops.append(-b_inj)
    tops.append(-b_inj - 0.01)              # to set plot properly
 
    # create hydraulic unit objects
    aquifers = Aquifers(np.array(K_aqf), np.array(b_aqf), np.array(z))
    aquitards = Aquitards(np.array(K_aqt), np.array(b_aqt))
 
    return aquifers, aquitards, tops
 
def PlotSection(tops):
    # draw vertical section showing aquifer and aquitard delineations
    plt.figure(0)
    clr = ['Peru', 'DeepSkyBlue']           # colors for aquitard, aquifer
    for (i, layer) in enumerate(tops):
        plt.bar(1., layer, color = clr[i%2])
        f = plt.gca()
        f.axes.get_xaxis().set_visible(False)
    plt.ylabel('Elev. Above Datum')
    aquifer_patch = mpatches.Patch(color='DeepSkyBlue', label='Aquifer')
    aquitard_patch = mpatches.Patch(color='Peru', label='Aquitard')
    plt.legend(handles=[aquifer_patch, aquitard_patch])
    plt.show()
 
def ProcessVertProfile(x, y, num_aqf, aquifer_layer, aquifers, well, V, Z):
    # draw vertical head change profile
    plt.figure(1)
    dh = np.zeros(num_aqf, float)
    for layer in aquifer_layer:
        for pump in well:
            dh[layer] = +pump.HeadChange(x, y, layer, num_aqf, aquifers, V, Z)
    vertical = {'layer': aquifer_layer, 'z': aquifers.z, 'b': aquifers.b, 'dh': dh}
    vertical_df = pd.DataFrame(vertical)
    vertical_df.to_csv('vertical.csv')
    plt.semilogx(vertical_df['dh'], vertical_df['z'])
    plt.xlabel('Head Change')
    plt.ylabel('Elev. Above Datum')
    plt.show()
 
def PlotLayerProfiles(profiles_df, aquifer_layer, x):
    # draw radial drawdown profiles for all aquifer layers
    plt.figure(1)
    for layer in aquifer_layer:
        subset_df = profiles_df[profiles_df['layer']==layer]
        plt.plot(subset_df['x'], subset_df['dh'],
            label='aquifer ' + str(layer))
    plt.legend(loc='lower right')
    plt.xlim(0., x.max())
    plt.xlabel('Radial Distance')
    plt.ylabel('Head Change')
    plt.show()    
 
def ProcessLayerProfiles(num_aqf, aquifer_layer, aquifers, well, V, Z):
    # process head changes along x-transect; write to file & return data frame
    x = np.linspace(100., 5000., 50)       # example transect
    y = 0.
    for (i, xp) in enumerate(x):
        dh = np.zeros(num_aqf, float)
        for layer in aquifer_layer:
            for pump in well:
                dh[layer] = +pump.HeadChange(xp, y, layer, num_aqf, aquifers, V, Z)
        profiles = {'layer': aquifer_layer, 'x': np.zeros(num_aqf)+xp,
            'y': np.zeros(num_aqf)+y, 'z': aquifers.z, 'b': aquifers.b,
            'dh': dh}
        if not i: profiles_df = pd.DataFrame(profiles)
        else:
            new_profiles_df = pd.DataFrame(profiles)
            profiles_df = pd.concat([profiles_df, new_profiles_df], axis=0)
    profiles_df.to_csv('profiles.csv')
    return profiles_df, x
 
###################################
#
# script
#
###################################
 
def Hemker(mode):
 
    # specifcy aquifer-aquitard structure
    if mode == 0:                 # example strata
        K_aqf = np.array([40., 50., 33.333, 40.])
        b_aqf = np.array([50., 30., 15., 50.])
        K_aqt = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        b_aqt = np.array([10., 15., 10., 40., 200.])
        z_aqf, tops = GetLayers(b_aqf, b_aqt)
        aquifers = Aquifers(K_aqf, b_aqf, z_aqf)
        aquitards = Aquitards(K_aqt, b_aqt)
    else:                       # create stochastic strata (example parameters)
        section = 2000.                 # total thickness
        num_contacts = 50              # must be an even number
        K_aqf0 = 30.
        K_aqt0 = 0.002
        K_inj = 0.98
        b_inj = 75.
        aquifers, aquitards, tops = Strat(section, num_contacts,
                                          K_aqf0, K_aqt0, K_inj, b_inj)
    PlotSection(tops)           # visualize aquifer-aquitard section
 
    # well properties
    num_aqf = len(aquifers.K)
    well = []
    if mode == 0:
        well.append(Well(0., 0., -10000., np.array([0, 1, 0, 0]), aquifers))   # example
    else:
        Qw = 18096.                         # example
        screen_list = np.zeros(num_aqf, int)
        screen_list[-1] = 1
        well.append(Well(0., 0., Qw, screen_list, aquifers))
 
    # set up Hemker matrix
    A = HemkerMatrix(aquifers, aquitards)
    print('Set up matrix for Hemker solution.')
 
    # solve eigenvalue problem
    w, V, Z = Eigen(A)
    aquifers.Leakance(w)
    print('Solved eigenvalue problem.')
 
    # model drawdown
    aquifer_layer = np.arange(0, num_aqf, 1)
    if mode == 0:
        profiles_df, x =  ProcessLayerProfiles(num_aqf, aquifer_layer,
            aquifers, well, V, Z)           # create output as a dataframe
        PlotLayerProfiles(profiles_df, aquifer_layer, x) # plot profiles over x
    else:
        # plot head, per layer, as a function of depth
        x = 5.      # example
        y = 0.
        ProcessVertProfile(x, y, num_aqf, aquifer_layer, aquifers, well, V, Z)
 
    print('Done.')
 
### run script ###
 
# mode==0 --> run problem with specified layer properties;
#   generate drawdown curves for every aquifer
# mode==1 --> generate many aquifer-aquitard zones stochastically;
#   generate section diagram and head change profile with depth for given r
 
mode = 1
Hemker(mode)
