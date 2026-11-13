import constants
import kinematics
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d

def dsigdlnEovR_oscstrength(T, lnEovR,OscStrengthFunction,M=constants.mT*2):
    EovR=np.exp(lnEovR)
    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, M))
#    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, M))
    ln_Ka2_max=0

    return (constants.NormConst *constants.R / (T / constants.R) * OscStrengthFunction(EovR*constants.R) * (ln_Ka2_max-ln_Ka2_min))

def sig_oscstrength(T,OscStrengthFunction,M=constants.mT*2):
    lnEovR_min = np.log(min(OscStrengthFunction.x)/constants.R)
    lnEovR_max = np.log(max(OscStrengthFunction.x)/constants.R)

    def ToInt(lnEovR):
        return (dsigdlnEovR_oscstrength(T, lnEovR,OscStrengthFunction,M))

    return (quad(ToInt, lnEovR_min, lnEovR_max)[0])

def SampleKinematics(T, OscStrengthFunction, M, rng=None):
    rng = rng or np.random.default_rng()
    lnEovR_min = np.log(min(OscStrengthFunction.x)/constants.R)
    lnEovR_max = np.log(max(OscStrengthFunction.x)/constants.R)
    lnErange=np.linspace(lnEovR_min,lnEovR_max,1000)
    SampleFunction=interp1d(np.cumsum(OscStrengthFunction(np.exp(lnErange)*constants.R))/sum(OscStrengthFunction(np.exp(lnErange)*constants.R)),lnErange,bounds_error=False,fill_value=0)

    lnEovR_sample=SampleFunction(rng.random())
    EovR=np.exp(lnEovR_sample)

    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, M))
    #ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, M))
    ln_Ka2_max=0
    ln_ka2_sample=rng.random()*(ln_Ka2_max-ln_Ka2_min)+ln_Ka2_min
    Ka2=np.exp(ln_ka2_sample)

    dE=EovR*constants.R

    dT = kinematics.getDeltaT(Ka2, dE)
    Theta = np.arccos(kinematics.getCosTheta(Ka2, T, dE))

    return dT,Theta


def GetKinematicFunction(filename,M):
    f = open(filename, 'r')
    lines = np.array(f.readlines())
    f.close()
    linesplit = np.argwhere(["-----------------------------" in l for l in lines])[0][0]
    dat = pd.read_csv(filename, skiprows=linesplit + 1, delimiter='\t', names=['E', 'dfdE'],
                          dtype={'E': float, 'dfdE': float}, skipfooter=2)
    Interp = interp1d(dat['E'], dat['dfdE'], fill_value=0, bounds_error=False, kind='linear')

    def ReturnFunction(T):
        dT_i, dTh_i = SampleKinematics(T, Interp, M)
        return dT_i, dTh_i

    return ReturnFunction


def GetBetheBornCrossSectionFunction(M2, C_nr):

    def sigma_bb(T_eV, M2, C_nr, z=1.0):
        b2 = 1.0 -  (1.0 + np.asarray(T_eV, float) / constants.me)**-2
        C_rel = C_nr + M2 * np.log(constants.me / (2.0 * constants.R))
        pref = 8.0 * np.pi * z ** 2 / (constants.me * b2 / constants.R)
        return pref * (M2 * (np.log(b2 / (1.0 - b2)) - b2) + C_rel)

    def ReturnFunction(T):
        return(sigma_bb(T, M2, C_nr))

    return ReturnFunction
