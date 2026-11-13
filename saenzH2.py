import pylab as plt
import numpy as np
import pandas as pd
import kinematics

from scipy.integrate import quad
import constants
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator

gosd_ex     = pd.read_csv("./Data/gosd_h2_R1p4.dat",delimiter=r'\s+',names=['K','E','GOSD'])
gosd_simple = pd.read_csv("./Data/gosd_h2_R1p4.dat",delimiter=r'\s+',names=['K','E','GOSD'])



logEovRs=np.log(gosd_simple.E*constants.Hartree/constants.R)

lnEovRMin=min(logEovRs)
lnEovRMax=max(logEovRs)
lnKa2Min=np.log(min(gosd_simple.K**2))
lnKa2Max=np.log(max(gosd_simple.K**2))


# This constructs an interpolator that also extrapolates linearly to (slightly below) the ionization threshold
ExtrapTo=15/constants.Hartree

MinE=sorted(gosd_simple.E.unique())[0]
SecE=sorted(gosd_simple.E.unique())[1]

GOSD_MinE=np.array(gosd_simple[gosd_simple.E==MinE].GOSD)
GOSD_SecE=np.array(gosd_simple[gosd_simple.E==SecE].GOSD)
Ks=gosd_simple[gosd_simple.E==MinE].K
Extrap_GOSD=GOSD_MinE+(GOSD_MinE-GOSD_SecE)*(MinE-ExtrapTo)/(SecE-MinE)
for i in range(len(Ks)):
    gosd_ex.loc["New"+str(i)]=([Ks.iloc[i],ExtrapTo,Extrap_GOSD[i]])

Points_ex     = np.array([gosd_ex.E,gosd_ex.K]).T
df_dE_ex      = LinearNDInterpolator(Points_ex,gosd_ex.GOSD)

Points_simple = np.array([gosd_simple.E, gosd_simple.K]).T
df_dE_simple  = LinearNDInterpolator(Points_simple, gosd_simple.GOSD)


def dsig_dlnEovR_dlnKa2(T,lnEovR,lnKa2,interpr):
    return interpr(lnEovR,lnKa2)/(T/constants.R)*constants.NormConst

def dsig_dlnEovR(T,lnEovR,interpr):
    def ToInt(lnKa2):
        return dsig_dlnEovR_dlnKa2(T,lnEovR,lnKa2,interpr)
    ln_Ka2_min = np.log(kinematics.Ka2Min(np.exp(lnEovR)*constants.R, T, 2*constants.mT/3))
    ln_Ka2_max = np.log(kinematics.Ka2Max(np.exp(lnEovR)*constants.R, T, 2*constants.mT/3))
    return quad(ToInt,ln_Ka2_min,ln_Ka2_max)[0]

def sig(T,interpr,lnEovRMin=lnEovRMin,lnEovRMax=lnEovRMax):
    def ToInt(lnEovR):
        return dsig_dlnEovR(T,lnEovR,interpr)
    return quad(ToInt,lnEovRMin,lnEovRMax)[0]
