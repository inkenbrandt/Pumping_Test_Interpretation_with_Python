#####################################################################
# pumpingtest.py
# aquifer test analysis by various methods
# by Walt McNab
#####################################################################

import numpy as np
# from numpy import *
from scipy.integrate import quad
from scipy.integrate import odeint
# from scipy.special import *
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



class Hantush:  # Hantush and Jacob (1955) solution

    def __init__(self, d, t, Q, T, K, b, Ss, Kc, bc):
        self.bc = bc
        self.d = d
        self.Ss = Ss
        self.t = t
        self.K = K
        self.b = b
        self.Kc = Kc
        self.Q = Q
        self.B = self.calc_B
        self.drawdown

    def calc_B(self):
        return np.sqrt(self.bc * self.K * self.b / self.Kc)

    def integrand(self, y):
        # integral term for the Hantush well function
        x = np.exp(-y - self.d ** 2 / (4. * self.B ** 2 * y)) / y
        return x

    def W(self, u):
        # Hantush well function
        x = quad(self.integrand, u, np.inf)[0]
        return x

    def drawdown(self):
        u = self.d ** 2 * self.Ss / (4 * self.K * self.t)
        s = -self.Q / (4 * np.pi * self.K * self.b) * self.W(u)
        return s


class ShortStorage:  # Hantush (1960) solution for leaky aquifer with aquitard storage (short-term)

    def __init__(self, d, t, Q, T, K, b, Ss, Kc, Ssc):
        self.beta = np.sqrt(Kc * Ssc / (K * Ss)) * 4.0 * d / b
        self.Ss = Ss
        self.Q = Q
        self.t = t
        self.drawdown

    def integrand(self, y, u):
        # integral term for the Hantush well function
        x = scipy.special.erfc(self.beta * np.sqrt(u) / np.sqrt(y * (y - u))) * np.exp(-y) / y
        return x

    def H(self, u):
        # Hantush modified well function
        x = quad(self.integrand, u, np.inf, args=(u))[0]
        return x

    def drawdown(self):
        u = self.d ** 2 * self.Ss / (4 * self.K * self.t)
        s = -self.Q / (4 * np.pi * self.K * self.b) * self.H(u)
        return s


class Theis:  # Theis (1935) solution

    def __init__(self, d, t, Q, T, K, b, Ss, Sy, mode=0):
        self.mode = mode
        self.Sy = Sy
        self.Q = Q
        self.d = d
        self.Ss = Ss
        self.K = K
        self.b = b
        self.t = t
        self.drawdown

    def W(self, u):
        # Theis well function
        return scipy.special.expn(1, u)

    def drawdown(self):
        if self.mode == 0:  # confined aquifer
            u = self.d ** 2 * self.Ss / (4 * self.K * self.t)
            s = -self.Q / (4 * np.pi * self.K * self.b) * self.W(u)
        else:  # unconfined aquifer (assuming ~ constant saturated thickness)
            u = self.d ** 2 * self.Sy / (4 * self.K * self.b * self.t)
            s = -self.Q / (4 * np.pi * self.K * self.b) * self.W(u)
        return s


class OldTheis:
    # adapted from: https://github.com/Applied-Groundwater-Modeling-2nd-Ed/Chapter_3_problems-1
    def __init__(self, d, t, Q, T, K, b, S):


    def well_function(self, u):
        return scipy.special.exp1(u)

    def theis(self, Q, t):
        u = self.d ** 2 * self.S / 4. / (self.K * self.b) / self.t
        s = self.Q / 4. / np.pi / (self.K * self.b) * self.well_function(u)
        return s


class MOL:  # numerical (method-of-lines) solution for an unconfined aquifer

    def __init__(self, aquifer, well):
        self.aquifer = aquifer
        self.well = well
        self.N = 70  # default number of radial grid cells
        self.rFace = self.Gridder()  # array of grid cell interface radii
        self.r = 0.5 * self.rFace[1:] + 0.5 * self.rFace[:-1]  # radius of node point associated with each cell
        self.r = np.insert(self.r, 0, self.well.r)  # cell representing well
        self.A = np.pi * (
                    self.rFace[1:] ** 2 - self.rFace[:-1] ** 2)  # base areas associated with individual grid cells
        self.A = np.insert(self.A, 0, np.pi * self.rFace[0] ** 2)
        self.Sy = np.zeros(self.N, float) + aquifer.Sy  # assign storage coefficient of 1.0 to wellbore cell
        self.Sy = np.insert(self.Sy, 0, 1.0)
        self.S = np.zeros(self.N, float) + aquifer.S
        self.S = np.insert(self.S, 0, 1.0)

    def Gridder(self):
        # generate radial grid
        rb = self.aquifer.b * 100.  # set fixed boundary condition = 10X the available drawdown
        index = np.arange(0, self.N + 1, 1)
        f = 10. ** (np.log10((rb / self.well.r)) / self.N)  # sequential scaling factor
        r = self.well.r * f ** index
        return r

    def Dupuit(self, h, t):
        # ordinary differential equations (volumetric balance for water) for grid cells; variable saturated thickness
        J = 2. * np.pi * self.aquifer.K * self.rFace[:-1] * (0.5 * h[1:] + 0.5 * h[:-1]) * (h[1:] - h[:-1]) / (
                    self.r[1:] - self.r[:-1])
        J = np.insert(J, 0, -self.well.Q)
        J = np.append(J, 2. * np.pi * self.aquifer.K * self.rFace[-1] * (0.5 * h[-1] + 0.5 * self.aquifer.b)
                      * (self.aquifer.b - h[-1]) / (
                                  self.rFace[-1] - self.r[-1]))  # append flux from across exterior boundary
        dhdt = (J[1:] - J[:-1]) / (self.A * self.Sy)
        return dhdt

    def Theis(self, h, t):
        # ordinary differential equations (volumetric balance for water) for grid cells; fixed saturated thickness
        J = 2. * np.pi * self.aquifer.K * self.rFace[:-1] * self.aquifer.b * (h[1:] - h[:-1]) / (
                    self.r[1:] - self.r[:-1])
        J = np.insert(J, 0, -self.well.Q)  # express pumping as extraction from well
        J = np.append(J, 2. * np.pi * self.aquifer.K * self.rFace[-1] * self.aquifer.b
                      * (self.aquifer.b - h[-1]) / (
                                  self.rFace[-1] - self.r[-1]))  # append flux from across exterior boundary
        dhdt = (J[1:] - J[:-1]) / (self.A * self.S)
        return dhdt

    def Drawdown(self, mode):
        # solve the transient unconfined aquifer test problem using the numerical method-of-lines
        h = np.zeros(self.N + 1, float) + self.aquifer.b
        if mode == 0:
            h_t = odeint(self.Dupuit, h, self.well.tArray)
        else:
            h_t = odeint(self.Theis, h, self.well.tArray)
        h_t = np.transpose(h_t)
        s = self.aquifer.b - h_t[0]  # drawdown vector for cell representing well bore
        return s
