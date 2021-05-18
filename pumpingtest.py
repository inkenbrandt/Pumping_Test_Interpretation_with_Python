#####################################################################
# pumpingtest.py
# aquifer test analysis by various methods
# by Walt McNab
#####################################################################

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.integrate import odeint
import scipy.special


class DataSet(object):

    def __init__(self, df, t="time", s="drawdown"):
        # pumping test data (time and drawdown arrays)
        self.t = df[t]
        self.s = df[s]


class Aquifer:

    def __init__(self):
        # aquifer characteristics
        lineInput = []
        inputFile = open('aquifer.txt', 'r')
        for line in inputFile: lineInput.append(line.split())
        inputFile.close()
        self.K = float(lineInput[0][1])  # aquifer properties
        self.Ss = float(lineInput[1][1])
        self.Sy = float(lineInput[2][1])
        self.b = float(lineInput[3][1])  # used as saturated thickness for unconfined aquifer
        self.bc = float(lineInput[4][1])
        self.Kc = float(lineInput[5][1])  # 'c' refers to clay/aquitard
        self.Ssc = float(lineInput[6][1])
        self.S = self.Ss * self.b  # derive storage coefficient from specific storage
        print('Read aquifer characteristics.')

    def WriteValues(self):
        # update parameter file with current values
        output_file = open('aquifer.txt', 'w')
        output_file.writelines(['K', '\t', str(self.K), '\n'])
        output_file.writelines(['Ss', '\t', str(self.Ss), '\n'])
        output_file.writelines(['Sy', '\t', str(self.Sy), '\n'])
        output_file.writelines(['b', '\t', str(self.b), '\n'])
        output_file.writelines(['bc', '\t', str(self.bc), '\n'])
        output_file.writelines(['Kc', '\t', str(self.Kc), '\n'])
        output_file.writelines(['Ssc', '\t', str(self.Ssc), '\n'])
        output_file.close()


class Well:

    def __init__(self, xcoord, ycoord, rad=6, distanceunits="m",
                 radunits="in", pumpdf=None, pumpunits="gpm",
                 obsdf=None, depthunits="ft"):

        # well radius; assume radial distance for monitoring drawdown
        self.s = None
        self.t_obs = None
        self.t_pump = None
        self.Q = None
        self.process_pump(pumpdf, pumpunits)
        self.process_obs(obsdf, depthunits)
        self.m_conversions = {"in": 0.0254, "ft": 0.3048, "cm": 0.01,
                              "m": 1.0, "km": 1000, "miles": 1609.34}
        self.r = rad * self.m_conversions.get(radunits, 1.0)
        self.xd = xcoord * self.m_conversions.get(distanceunits, 1.0)
        self.yd = ycoord * self.m_conversions.get(distanceunits, 1.0)
        #self.tArray = np.logspace(np.log10(0.0001), np.log10(10000), num=60, endpoint=True)

    def process_obs(self, obsdf, depthunits):
        if obsdf:
            self.s = obsdf.drawdown * self.m_conversions.get(depthunits)
            self.t_obs = obsdf.index.to_series().diff().fillna(0).astype('timedelta64[s]').astype('int')

    def process_pump(self, pumpdf, pumpunits):
        if pumpdf:
            if pumpunits == "gpm":
                pumpdf.Q = pumpdf.Q * 6.309e-5
            elif pumpunits == "cfs":
                pumpdf.Q = pumpdf.Q * 0.0283168
            elif pumpunits == "liters per second":
                pumpdf.Q = pumpdf.Q * 0.001

            self.t_pump = pumpdf.index.to_series().diff().fillna(0).astype('timedelta64[s]').astype('int')
            self.Q = pumpdf.Q
        else:
            pass


def Hantush(d, t, Q, T=None, K=None, b=30, Ss=0.001, Kc=0.001, bc=10):
    """Hantush and Jacob (1955) solution

    :param d: distance from pumping well (m)
    :param t: time since pumping started (min)
    :param Q: pumping rate (m3/min)
    :param b: aquifer thickness (m); defaults to 30m
    :param Ss: aquifer Specific Storage (unitless); defaults to 0.001
    :param T: transmissivity (m2/day); defaults to None
    :param K: aquifer hydraulic conductivity (m/day); optional; defaults to None
    :param Kc: confining layer hydraulic conductivity (m/day)
    :param bc: confining layer thickness (m)
    :returns drawdown
    """
    d = abs(d)

    if t <= 0 or d == 0:
        s = 0
    else:
        if T:
            K = T/b
        else:
            T = K*b

        B = np.sqrt(bc * K * b / Kc)

        integrand = lambda y: np.exp(-y - d ** 2 / (4. * B ** 2 * y)) / y
        W = lambda u: scipy.integrate.quad(integrand, u, np.inf)[0]

        u = d ** 2 * Ss / (4 * K * t)
        s = -Q / (4 * np.pi * K * b) * W(u)
    return s


def ShortStorage(d, t, Q, T=30., K=1., b=30., Ss=0.001, Kc=1, Ssc=0.001):
    """Hantush (1960) solution for leaky aquifer with aquitard storage (short-term)

    :param d: distance from pumping well (m)
    :param t: time since pumping started (min)
    :param Q: pumping rate (m3/min)
    :param T: transmissivity (m2/day); defaults to None
    :param K: aquifer hydraulic conductivity (m/day); optional; defaults to None
    :param b: aquifer thickness (m); defaults to 30m
    :param Ss: aquifer Specific Storage (unitless); defaults to 0.001
    :param Kc: confining layer hydraulic conductivity (m/day)
    :param Ssc: confining layer specific storage (unitless); defaults to 0.001
    :returns: drawdown
    """
    d = abs(d)

    if t <= 0 or d == 0:
        s = 0
    else:
        if T:
            K = T / b
        else:
            T = K * b

        beta = np.sqrt(Kc * Ssc / (K * Ss)) * 4.0 * d / b
        u = d ** 2 * Ss / (4 * K * t)

        # integral term for the Hantush well function
        integrand = lambda y, u: scipy.special.erfc(beta * np.sqrt(u) / np.sqrt(y * (y - u))) * np.exp(-y) / y

        H = lambda u: scipy.integrate.quad(integrand, u, np.inf, args=(u))[0]

        s = -Q / (4 * np.pi * K * b) * H(u)
    return s

def Theis(d, t, Q, T=30., K=1., b=30., Ss=0.001, Sy=0.1, mode=0):
    """Theis (1935) solution

    :param d: distance from pumping well (m)
    :param t: time since pumping started (min)
    :param Q: pumping rate (m3/min)
    :param T: transmissivity (m2/day); defaults to None
    :param K: aquifer hydraulic conductivity (m/day); optional; defaults to None
    :param b: aquifer thickness (m); defaults to 30m
    :param Ss: aquifer Specific Storage (unitless); defaults to 0.001
    :param Sy: aquifer specific yield; defaults to 0.1
    :param mode: confinement state; 0 = confined; 1 = unconfined; defaults to 0

    :returns: drawdown
    """
    d = abs(d)

    if t <= 0 or d == 0:
        s = 0
    else:
        if T:
            K = T/b
        else:
            T = K*b

        if mode == 0:  # confined aquifer
            u = d ** 2 * Ss / (4 * K * t)
        else:  # unconfined aquifer (assuming ~ constant saturated thickness)
            u = d ** 2 * Sy / (4 * K * b * t)

        s = -Q / (4. * np.pi * K * b) * scipy.special.exp1(u)
    return s


def variable_q(qdf, d, method, tend=None, **kwargs):
    if tend:
        pass
    else:
        tend = qdf.last_valid_index() * 2

    qdf = qdf.sort_index()

    # if there are no flow measurements at 0 minutes add one with a flow of 0
    if qdf.index[0] != 0:
        qdf.loc[0, 'flow'] = 0
        qdf = qdf.sort_index()

    # get change in flow for each step and make initial change the initial flow instead of None
    qdf['dQ'] = qdf.flow.diff().fillna(qdf.iloc[0]['flow'])

    # create empty dataframe with regular time steps that spans the defined time range
    sdf = pd.DataFrame(index=np.arange(0, tend, 1), columns=qdf.index.values)

    # process the drawdown at each time step then shift to pump change time
    for i in qdf.index:
        sdf[i] = sdf.apply(lambda x: method(d, x.name, Q=-1 * qdf.loc[i, 'dQ'], **kwargs), 1).shift(i)

    # add all the pumping steps together
    sdf['all'] = sdf.sum(axis=1)
    return sdf['all']

class MOL:  # numerical (method-of-lines) solution for an unconfined aquifer

    def __init__(self, d, t, Q, T, K, b, Sy, S):
        #self.aquifer = aquifer
        #self.well = well
        self.d = d
        self.Q = Q
        self.S = S
        self.Sy = Sy
        self.T = T
        self.K = K
        self.b = b
        self.t = t
        self.N = 70  # default number of radial grid cells
        self.rFace = self.Gridder()  # array of grid cell interface radii
        self.r = 0.5 * self.rFace[1:] + 0.5 * self.rFace[:-1]  # radius of node point associated with each cell
        self.r = np.insert(self.r, 0, self.d)  # cell representing well
        self.A = np.pi * (
                    self.rFace[1:] ** 2 - self.rFace[:-1] ** 2)  # base areas associated with individual grid cells
        self.A = np.insert(self.A, 0, np.pi * self.rFace[0] ** 2)
        self.Sy = np.zeros(self.N, float) + self.Sy  # assign storage coefficient of 1.0 to wellbore cell
        self.Sy = np.insert(self.Sy, 0, 1.0)
        self.S = np.zeros(self.N, float) + self.S
        self.S = np.insert(self.S, 0, 1.0)
        self.drawdown

    def Gridder(self):
        # generate radial grid
        rb = self.b * 100.  # set fixed boundary condition = 10X the available drawdown
        index = np.arange(0, self.N + 1, 1)
        f = 10. ** (np.log10((rb / self.d)) / self.N)  # sequential scaling factor
        r = self.d * f ** index
        return r

    def Dupuit(self, h, t):
        # ordinary differential equations (volumetric balance for water) for grid cells; variable saturated thickness
        J = 2. * np.pi * self.K * self.rFace[:-1] * (0.5 * h[1:] + 0.5 * h[:-1]) * (h[1:] - h[:-1]) / (
                    self.r[1:] - self.r[:-1])
        J = np.insert(J, 0, -self.Q)
        J = np.append(J, 2. * np.pi * self.K * self.rFace[-1] * (0.5 * h[-1] + 0.5 * self.b)
                      * (self.b - h[-1]) / (
                                  self.rFace[-1] - self.r[-1]))  # append flux from across exterior boundary
        dhdt = (J[1:] - J[:-1]) / (self.A * self.Sy)
        return dhdt

    def Theis(self, h, t):
        # ordinary differential equations (volumetric balance for water) for grid cells; fixed saturated thickness
        J = 2. * np.pi * self.K * self.rFace[:-1] * self.b * (h[1:] - h[:-1]) / (
                    self.r[1:] - self.r[:-1])
        J = np.insert(J, 0, -self.Q)  # express pumping as extraction from well
        J = np.append(J, 2. * np.pi * self.K * self.rFace[-1] * self.b
                      * (self.b - h[-1]) / (
                                  self.rFace[-1] - self.d[-1]))  # append flux from across exterior boundary
        dhdt = (J[1:] - J[:-1]) / (self.A * self.S)
        return dhdt

    def drawdown(self, mode):
        # solve the transient unconfined aquifer test problem using the numerical method-of-lines
        h = np.zeros(self.N + 1, float) + self.b
        if mode == 0:
            h_t = odeint(self.Dupuit, h, self.t)
        else:
            h_t = odeint(self.Theis, h, self.t)
        h_t = np.transpose(h_t)
        s = self.b - h_t[0]  # drawdown vector for cell representing well bore
        return s
