import numpy as  np
import kinematics
import constants
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
def MakeSampleDicitonary(T, movM, ns=range(2, 10)):
    DiffXSSampleFunction = {}

    DiffXS_LogKa2 = {}
    for n in ns:
        q2min = kinematics.Ka2Min(kinematics.En(n), T, constants.movM)
        q2max = kinematics.Ka2Max(kinematics.En(n), T, constants.movM)
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
def dsigdEovR_continuum(T, EovR):
    def ToInt(lnKa2):
        return (dfdEovR_ln_continuum(EovR, lnKa2))

    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * R, T, constants.movM))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * R, T, constants.movM))
    return (constants.NormConst / ((T / constants.R) * (EovR)) * quad(ToInt, ln_Ka2_min, ln_Ka2_max)[0])


# Differential cross section integrated over Ka2 to continuum at ln(E/R) -
#  this one gives better convergence in the integral.
def dsigdlnEovR_continuum(T, lnEovR):
    EovR = np.exp(lnEovR)

    def ToInt(lnKa2):
        return (dfdEovR_ln_continuum(EovR, lnKa2))

    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, constants.movM))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, constants.movM))
    return (constants.NormConst / (T / constants.R) * quad(ToInt, ln_Ka2_min, ln_Ka2_max)[0])


# Total cross section into the continuum, integrated over E/R
def sig_continuum(T):
    EovR_min = np.log(1.00001)
    EovR_max = np.log(T / constants.R - 1)

    def ToInt(EovR):
        return (dsigdlnEovR_continuum(T, EovR))

    return (quad(ToInt, EovR_min, EovR_max)[0])


# Cross section between kinematic limits
def DistributionDraw(EovR, lnKa2, T=20e3, movM=constants.movM):
    ln_Ka2_min = np.log(kinematics.Ka2Min(EovR * constants.R, T, constants.movM))
    ln_Ka2_max = np.log(kinematics.Ka2Max(EovR * constants.R, T, constants.movM))
    returnval = (lnKa2 > ln_Ka2_min) * (lnKa2 < ln_Ka2_max) * (EovR > 1.0) * (
                constants.NormConst / (T / constants.R) * dfdEovR_ln_continuum(EovR, lnKa2) / (EovR))
    return returnval

def SampleContinuum(T,Distn,Emin=1.0001,Emax=10,logKmin=-9,logKmax=6,samples=100000,bins=100):
    draws=sample_2d_array(Distn,samples)
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

def MakeSampleFunction_elastic(T,movM):

    q2min=kinematics.Ka2Min(0,T,constants.movM)+1e-10
    q2max=kinematics.Ka2Max(0,T,constants.movM)
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


#=========================================
# Utility functions
#=========================================

#Function to randomly sample from a 2d histogram (Thanks Claude for this one!)
def sample_2d_array(A, n_samples, extent=None, rng=None):
    """
    A: 2D array of weights/intensities, shape (nrows, ncols) — as passed to imshow
    extent: optional (xmin, xmax, ymin, ymax), matching imshow's `extent` kwarg.
            If None, samples are returned as (col, row) pixel coordinates.
    """
    rng = rng or np.random.default_rng()

    A = np.asarray(A, dtype=float)
    A = np.clip(A, 0, None)  # guard against negative weights
    p = A.ravel()
    p /= p.sum()

    flat_idx = rng.choice(p.size, size=n_samples, p=p)
    row, col = np.unravel_index(flat_idx, A.shape)  # row ~ y, col ~ x

    # jitter within each pixel (assume pixel is a unit cell)
    row_j = row + rng.uniform(-0.5, 0.5, size=n_samples)
    col_j = col + rng.uniform(-0.5, 0.5, size=n_samples)

    if extent is None:
        return col_j, row_j  # pixel coordinates (x=col, y=row)

    xmin, xmax, ymin, ymax = extent
    nrows, ncols = A.shape
    x_samples = xmin + (col_j / ncols) * (xmax - xmin)
    # row 0 is typically the top, i.e. y = ymax, unless origin='lower'
    y_samples = ymax - (row_j / nrows) * (ymax - ymin)

    return x_samples, y_samples