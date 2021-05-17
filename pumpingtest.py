#####################################################################
# pumpingtest.py
# aquifer test analysis by various methods
# by Walt McNab
#####################################################################

import numpy as np
#from numpy import *
from scipy.integrate import quad
from scipy.integrate import odeint
#from scipy.special import *
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
        inputFile = open('aquifer.txt','r')
        for line in inputFile: lineInput.append(line.split())
        inputFile.close()
        self.K = float(lineInput[0][1])     # aquifer properties
        self.Ss = float(lineInput[1][1])
        self.Sy = float(lineInput[2][1])
        self.b = float(lineInput[3][1])     # used as saturated thickness for unconfined aquifer
        self.bc = float(lineInput[4][1])        
        self.Kc = float(lineInput[5][1])    # 'c' refers to clay/aquitard
        self.Ssc = float(lineInput[6][1])
        self.S = self.Ss * self.b           # derive storage coefficient from specific storage
        print('Read aquifer characteristics.')

    def WriteValues(self):
        # update parameter file with current values
        output_file = open('aquifer.txt','w')
        output_file.writelines(['K', '\t', str(self.K),'\n'])
        output_file.writelines(['Ss', '\t', str(self.Ss), '\n'])
        output_file.writelines(['Sy', '\t', str(self.Sy), '\n'])
        output_file.writelines(['b', '\t', str(self.b), '\n'])
        output_file.writelines(['bc', '\t', str(self.bc), '\n'])        
        output_file.writelines(['Kc', '\t', str(self.Kc), '\n'])        
        output_file.writelines(['Ssc', '\t', str(self.Ssc), '\n'])
        output_file.close()         
        
        
class Well:
    
    def __init__(self, t0, tEnd):
        # well properties
        lineInput = []        
        inputFile = open('well.txt','r')
        for line in inputFile:
            lineInput.append(line.split())
        inputFile.close()
        self.r = float(lineInput[0][1])     # well radius; assume radial distance for monitoring drawdown
        self.Q = float(lineInput[1][1])     # pumping rate from well (negative value = extraction)
        self.tArray = np.logspace(np.log10(t0), np.log10(tEnd), num=60, endpoint=True)     # evaluation times

    def WriteValues(self):
        # update parameter file with current values
        output_file = open('well.txt','w')
        output_file.writelines(['r', '\t', str(self.r),'\n'])
        output_file.writelines(['Q', '\t', str(self.Q),'\n'])
        output_file.close()     

        
class Hantush:            # Hantush and Jacob (1955) solution

    def __init__(self, aquifer, well):
        self.B = np.sqrt(aquifer.bc*aquifer.K*aquifer.b/aquifer.Kc)
        self.aquifer = aquifer
        self.well = well

    def Integrand(self, y):
        # integral term for the Hantush well function
        x = np.exp(-y - self.well.r**2/(4.*self.B**2*y))/y
        return x

    def W(self, u):
        # Hantush well function
        x = quad(self.Integrand, u, np.inf)[0]
        return x
        
    def Drawdown(self):
        s = np.zeros(len(self.well.tArray), float)
        for i, t in enumerate(self.well.tArray):        
            u = self.well.r**2*self.aquifer.Ss/(4*self.aquifer.K*t)
            s[i] = -self.well.Q/(4*np.pi*self.aquifer.K*self.aquifer.b) * self.W(u)
        return s



        
class ShortStorage:            # Hantush (1960) solution for leaky aquifer with aquitard storage (short-term)

    def __init__(self, aquifer, well):
        self.beta = np.sqrt(aquifer.Kc*aquifer.Ssc/(aquifer.K*aquifer.Ss)) * 4.0*well.r/aquifer.b
        self.aquifer = aquifer
        self.well = well

    def Integrand(self, y, u):
        # integral term for the Hantush well function
        x = scipy.special.erfc(self.beta * np.sqrt(u)/np.sqrt(y*(y-u))) * np.exp(-y)/y
        return x

    def H(self, u):
        # Hantush modified well function
        x = quad(self.Integrand, u, np.inf, args=(u))[0]
        return x
        
    def Drawdown(self):
        s = np.zeros(len(self.well.tArray), float)
        for i, t in enumerate(self.well.tArray):        
            u = self.well.r**2*self.aquifer.Ss/(4*self.aquifer.K*t)
            s[i] = -self.well.Q/(4*np.pi*self.aquifer.K*self.aquifer.b) * self.H(u)
        return s        

        
class Theis:    # Theis (1935) solution

    def __init__(self, aquifer, well):
        self.aquifer = aquifer
        self.well = well
        
    def W(self, u):
        # Theis well function
        return scipy.special.expn(1, u)

    def Drawdown(self, mode):
        s = np.zeros(len(self.well.tArray), float)
        if mode == 0:       # confined aquifer
            for i, t in enumerate(self.well.tArray):    
                u = self.well.r**2 * self.aquifer.Ss/(4*self.aquifer.K*t)
                s[i] = -self.well.Q/(4*np.pi*self.aquifer.K*self.aquifer.b) * self.W(u)
        else:               # unconfined aquifer (assuming ~ constant saturated thickness)
            for i, t in enumerate(self.well.tArray):    
                u = self.well.r**2 * self.aquifer.Sy/(4*self.aquifer.K*self.aquifer.b*t)
                s[i] = -self.well.Q/(4*np.pi*self.aquifer.K*self.aquifer.b) * self.W(u)
        return s

class OldTheis:
    # adapted from: https://github.com/Applied-Groundwater-Modeling-2nd-Ed/Chapter_3_problems-1
    def __init__(self, aquifer, well):
        self.aquifer = aquifer
        self.well = well

    def well_function(self,u):
        return scipy.special.exp1(u)

    def theis(self, Q, T, S, r, t):
        u = self.well.r ** 2 * self.aquifer.S / 4. / self.aquifer.T / t
        s = Q / 4. / np.pi / self.aquifer.T * self.well_function(u)
        return s

class MOL:  # numerical (method-of-lines) solution for an unconfined aquifer
    
    def __init__(self, aquifer, well):
        self.aquifer = aquifer
        self.well = well
        self.N = 70                                                 # default number of radial grid cells       
        self.rFace = self.Gridder()                                 # array of grid cell interface radii
        self.r = 0.5*self.rFace[1:] + 0.5*self.rFace[:-1]           # radius of node point associated with each cell
        self.r = np.insert(self.r, 0, self.well.r)                     # cell representing well
        self.A = np.pi*(self.rFace[1:]**2 - self.rFace[:-1]**2)        # base areas associated with individual grid cells
        self.A = np.insert(self.A, 0, np.pi*self.rFace[0]**2)
        self.Sy = np.zeros(self.N, float) + aquifer.Sy                 # assign storage coefficient of 1.0 to wellbore cell
        self.Sy = np.insert(self.Sy, 0, 1.0)
        self.S = np.zeros(self.N, float) + aquifer.S
        self.S = np.insert(self.S, 0, 1.0)
    
    def Gridder(self):
        # generate radial grid
        rb = self.aquifer.b * 100.                   # set fixed boundary condition = 10X the available drawdown        
        index = np.arange(0, self.N+1, 1)
        f = 10.**(np.log10((rb/self.well.r))/self.N)   # sequential scaling factor
        r = self.well.r * f**index
        return r

    def Dupuit(self, h, t):
        # ordinary differential equations (volumetric balance for water) for grid cells; variable saturated thickness
        J = 2. * np.pi * self.aquifer.K * self.rFace[:-1] * (0.5*h[1:] + 0.5*h[:-1]) * (h[1:] - h[:-1]) / (self.r[1:] - self.r[:-1])
        J = np.insert(J, 0, -self.well.Q)
        J = np.append(J, 2.*np.pi*self.aquifer.K*self.rFace[-1]*(0.5*h[-1]+0.5*self.aquifer.b)
            *(self.aquifer.b-h[-1])/(self.rFace[-1]-self.r[-1]))            # append flux from across exterior boundary
        dhdt = (J[1:] - J[:-1]) / (self.A * self.Sy)
        return dhdt       
    
    def Theis(self, h, t):
        # ordinary differential equations (volumetric balance for water) for grid cells; fixed saturated thickness
        J = 2. * np.pi * self.aquifer.K * self.rFace[:-1] * self.aquifer.b * (h[1:] - h[:-1]) / (self.r[1:] - self.r[:-1])
        J = np.insert(J, 0, -self.well.Q)                                      # express pumping as extraction from well
        J = np.append(J, 2.*np.pi*self.aquifer.K*self.rFace[-1]*self.aquifer.b
            *(self.aquifer.b-h[-1])/(self.rFace[-1]-self.r[-1]))            # append flux from across exterior boundary
        dhdt = (J[1:] - J[:-1]) / (self.A * self.S)
        return dhdt 
    
    def Drawdown(self, mode):
        # solve the transient unconfined aquifer test problem using the numerical method-of-lines
        h = np.zeros(self.N+1,float) + self.aquifer.b
        if mode == 0: h_t = odeint(self.Dupuit, h, self.well.tArray)
        else: h_t = odeint(self.Theis, h, self.well.tArray)
        h_t = np.transpose(h_t)
        s = self.aquifer.b - h_t[0]         # drawdown vector for cell representing well bore
        return s      

