import constants
import numpy as np

# Excitation to nth discrete energy level
def En(n):
    return constants.R*(1-1/n**2)

# Minimum and maximum allowed Ka2 by kinematics
def Ka2Min(E,T,movM):
    #return 0.25*En(n)**2/(R*T)*(1+0.5*movM*En(n)/T)
    return 2*T/constants.R*constants.movM**-2*(1-0.5*constants.movM*E/T-(1-constants.movM*E/T)**0.5)

def Ka2Max(E,T,movM):
    #return 4*(T/R)*movM**-2*(1-0.5*movM*En(n)/T)
    return 2*T/constants.R*constants.movM**-2*(1-0.5*constants.movM*E/T+(1-constants.movM*E/T)**0.5)

# Get K in dimensions of eV from Ka2
def getK(Ka2):
    return(np.sqrt(Ka2)/constants.BohrRad*constants.hbarc)

# Get the deltaT given a Ka2
def getDeltaT(Ka2,E):
    return(Ka2/(constants.BohrRad/constants.hbarc)**2/(2*constants.mT) + E)

# Get the cos(theta) given a Ka2
def getCosTheta(Ka2,T,E):
    numerator=1-Ka2/(2*(T/constants.R)*(constants.movM)**-2)-0.5*constants.movM*E/T
    denominator=np.sqrt(1-constants.movM*E/T)
    return numerator/denominator

