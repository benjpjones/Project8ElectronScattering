import numpy as  np
import kinematics
import constants
import sampling
import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d

#========================
# Inelastic to discrete
#========================

# Bethes formula for oscillator sum to nth discrete level
def fn_ln_discrete(n, lnKa2):
    Ka = np.sqrt(np.exp(lnKa2))
    return 2 ** 8 * n ** 5 * (n ** 2 - 1) * (1 / 3 * (n ** 2 - 1) + (n * Ka) ** 2) * \
        ((n - 1) ** 2 + (n * Ka) ** 2) ** (n - 3) * ((n + 1) ** 2 + (n * Ka) ** 2) ** (-n - 3)


# Dfiferential cross section for discrete excitation
def dsigdLnKa_discrete(T, n, lnKa2):
    return (constants.NormConst / (T / constants.R * kinematics.En(n) / constants.R) * fn_ln_discrete(n, lnKa2))


# Make function to sample diff XS from
def MakeSampleDicitonary(T, M, ns=range(2, 10)):
    DiffXSSampleFunction = {}

    DiffXS_LogKa2 = {}
    for n in ns:
        q2min = kinematics.Ka2Min(kinematics.En(n), T, M)
        q2max = kinematics.Ka2Max(kinematics.En(n), T, M)
        logq2_limited = np.linspace(np.log(q2min), np.log(q2max), 100)
        DiffXS_LogKa2[n] = interp1d(logq2_limited, dsigdLnKa_discrete(T, n, logq2_limited))

    for n in DiffXS_LogKa2.keys():
        vars = np.linspace(DiffXS_LogKa2[n].x[0], DiffXS_LogKa2[n].x[-1], 1000)
        cumulative = np.cumsum(DiffXS_LogKa2[n](vars)) / sum(DiffXS_LogKa2[n](vars))
        cumulative[0] = 0
        interptosample = interp1d(cumulative, vars)
        DiffXSSampleFunction[n] = interptosample
    return DiffXSSampleFunction


def GetRandomKa2_discrete(n, SampleDictionary, size=1000):
    randomdraws = np.random.uniform(size=size)
    return (np.exp(SampleDictionary[n](randomdraws)))



#=================================
# Inelastic to continuum
#=================================


# Bethes formula for oscillator strength to contonium at energy E/R
def dfdEovR_ln_continuum(EovR, lnKa2):
    Ka2 = np.exp(lnKa2)
    Ka = np.sqrt(Ka2)
    kapa = np.sqrt(EovR - 1)
    numerator = 2 ** 7 * ((Ka2) + (1 / 3) * EovR) * EovR
    denominator = ((Ka + kapa) ** 2 + 1) ** 3 * ((Ka - kapa) ** 2 + 1) ** 3
    term1 = numerator / denominator
    term2 = (1 - np.exp(-2 * np.pi / (kapa))) ** -1
    term3 = np.exp(-2 / kapa * (np.arctan(2 * kapa / (Ka ** 2 - kapa ** 2 + 1)) % np.pi))
    return term1 * term2 * term3


# Differential cross section integrated over Ka2 to continuum at E/R
def dsigdEovR_continuum(T, EovR,M=constants.mT):
    def ToInt(lnKa2):
        return (dfdEovR_ln_continuum(EovR, lnKa2))

    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * R, T, M))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * R, T, M))
    return (constants.NormConst / ((T / constants.R) * (EovR)) * quad(ToInt, ln_Ka2_min, ln_Ka2_max)[0])


# Differential cross section integrated over Ka2 to continuum at ln(E/R) -
#  this one gives better convergence in the integral.
def dsigdlnEovR_continuum(T, lnEovR,M=constants.mT):
    EovR = np.exp(lnEovR)

    def ToInt(lnKa2):
        return (dfdEovR_ln_continuum(EovR, lnKa2))

    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, M))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, M))
    return (constants.NormConst / (T / constants.R) * quad(ToInt, ln_Ka2_min, ln_Ka2_max)[0])


# Total cross section into the continuum, integrated over E/R
def sig_continuum(T):
    lnEovR_min = np.log(1.00001)
    lnEovR_max = np.log(T / constants.R - 1)

    def ToInt(EovR):
        return (dsigdlnEovR_continuum(T, EovR))

    return (quad(ToInt, lnEovR_min, lnEovR_max)[0])


# Cross section between kinematic limits
def DistributionDraw(EovR, lnKa2, T=20e3, M=constants.mT):
    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, M))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, M))
    returnval = (lnKa2 > ln_Ka2_min) * (lnKa2 < ln_Ka2_max) * (EovR > 1.0) * (
                constants.NormConst / (T / constants.R) * dfdEovR_ln_continuum(EovR, lnKa2) / (EovR))
    return returnval

def SampleContinuum(T,Distn,Emin=1.0001,Emax=10,logKmin=-9,logKmax=6,samples=100000,bins=100):
    draws=sampling.sample_2d_array(Distn,samples)
    E=draws[0]/bins*(Emax-Emin)+Emin
    logK=draws[1]/bins*(logKmax-logKmin)+logKmin
    theta= np.arccos(kinematics.getCosTheta(np.exp(logK),T,E*constants.R))
    dT   = kinematics.getDeltaT(np.exp(logK),E*constants.R)
    return(theta,dT,E,logK)



#=================================
# Elastic
#=================================

def dsigdKshape_elastic(Ka2):
    return Ka2**-2*(1-1/(1+0.25*Ka2)**2)**2

def dsigdlnKshape_elastic(Ka2):
    return Ka2**-1*(1-1/(1+0.25*Ka2)**2)**2

# Total elastic cross section
def sig_elastic(T):
    TovR=T/constants.R
    return 4*np.pi*constants.BohrRad**2*(12+18*TovR+7*TovR**2)/(12*(1+TovR)**3)

def MakeSampleFunction_elastic(T,M):

    q2min=kinematics.Ka2Min(0,T,M)+1e-10
    q2max=kinematics.Ka2Max(0,T,M)
    logq2_limited=np.linspace(np.log(q2min),np.log(q2max),100)
    DiffXS_LogKa2=interp1d(logq2_limited,dsigdlnKshape_elastic(np.exp(logq2_limited)))
    vars=np.linspace(DiffXS_LogKa2.x[0],DiffXS_LogKa2.x[-1],1000)
    cumulative=np.cumsum(DiffXS_LogKa2(vars))/sum(DiffXS_LogKa2(vars))
    cumulative[0]=0
    interptosample=interp1d(cumulative, vars)
    DiffXSSampleFunction=interptosample
    return DiffXSSampleFunction

def GetRandomKa2_elastic(DiffXSSampleFunction,size=1000):  # in natural units
    randomdraws=np.random.uniform(size=size)
    return(np.exp(DiffXSSampleFunction(randomdraws)))




#============================================
# Make dictionaries to use for sampling codes
#============================================

def MakeTotalCrossSections(Ts=np.linspace(1e2,20e3,100)):

    TotalCrossSections={}

    #Discrete cross sections
    for n in range(2,10):
        sig=[]

        for T in Ts:
            ln_qmin2=np.log(kinematics.Ka2Min(kinematics.En(n),T,constants.mT))
            ln_qmax2=np.log(kinematics.Ka2Max(kinematics.En(n),T,constants.mT))
            def ToInt(lnKa2):
                return dsigdLnKa_discrete(T,n,lnKa2)
            sig.append(quad(ToInt,ln_qmin2,ln_qmax2)[0])
        TotalCrossSections["discrete_"+str(n)]=interp1d(Ts,sig)

    sigma_cont=[sig_continuum(T) for T in Ts]
    TotalCrossSections["continuum"]=interp1d(Ts,sigma_cont)

    TotalCrossSections["elastic"]=interp1d(Ts,sig_elastic(Ts))
    return TotalCrossSections

def MakeKinematicsFunctions():

    KinematicsFunctions={}

    # Set up grid for sampling double diff continuum cross section
    #  It will sample from a 2D histogram with E/R between Emin and
    #  Emax, logKa2min and logka2max, with dimensionality of bins
    #  in each direction.  Use the example in TechNotePlots to make
    #  sure this suitably spans the space where the cross section is
    #  significant, if in doubt.
    Emin = 1.00001;
    Emax = 10;
    logKa2min = -9;
    logKa2max = 6;
    bins = 100
    EE, KK = np.meshgrid(np.linspace(Emin, Emax, bins), np.linspace(logKa2min, logKa2max, bins))

    def ElasticSample(T):
        SampleFuncEl = MakeSampleFunction_elastic(T, constants.mT)
        Ka2 = GetRandomKa2_elastic(SampleFuncEl, 1)[0]
        dT = kinematics.getDeltaT(Ka2, 0)
        Theta = np.arccos(kinematics.getCosTheta(Ka2, T, 0))
        return (dT, Theta)
    KinematicsFunctions["elastic"]=ElasticSample

    def ContinuumSample(T):
        Distn_continuum = DistributionDraw(EE, KK, T, constants.mT)
        Theta, dT, E, logK = SampleContinuum(T, Distn_continuum, Emin, Emax, logKa2min, logKa2max, samples=1)
        Theta = Theta[0]
        dT = dT[0]
        return(dT,Theta)
    KinematicsFunctions["continuum"]=ContinuumSample

    def DiscreteSample(T,n):
        SampleDictInel = MakeSampleDicitonary(T, constants.mT, ns=range(2, 10))
        dE = kinematics.En(n)
        Ka2 = GetRandomKa2_discrete(n, SampleDictInel, 1)[0]
        dT = kinematics.getDeltaT(Ka2, dE)
        Theta = np.arccos(kinematics.getCosTheta(Ka2, T, dE))
        return(dT,Theta)
    KinematicsFunctions["discrete"]=DiscreteSample

    return KinematicsFunctions
