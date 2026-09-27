import constants
import kinematics
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

def dsigdlnEovR_oscstrength(T, lnEovR,OscStrengthFunction,M=constants.mT*2):
    EovR=np.exp(lnEovR)
    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, M))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, M))

    return (constants.NormConst / (T / constants.R) *OscStrengthFunction(EovR*constants.R) * (ln_Ka2_max-ln_Ka2_min))

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
    SampleFunction=interp1d(np.cumsum(OscStrengthFunction(np.exp(lnErange)*constants.R))/sum(OscStrengthFunction(np.exp(lnErange)*constants.R)),lnErange)
    lnEovR_sample=SampleFunction(rng.random())
    EovR=np.exp(lnEovR_sample)

    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, M))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, M))

    ln_ka2_sample=rng.random()*(ln_Ka2_max-ln_Ka2_min)+ln_Ka2_min
    Ka2=np.exp(ln_ka2_sample)
    dE=EovR*constants.R

    dT = kinematics.getDeltaT(Ka2, dE)
    Theta = np.arccos(kinematics.getCosTheta(Ka2, T, dE))

    return dT,Theta
