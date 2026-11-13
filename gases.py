import opticalapprox
import constants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


A0_CM = 5.29177210903e-9

def C_from_Mlnc(M2, M2_ln_c):
    """C_nr from the published product M2*ln(c_tot):  C_nr = M2 ln c_tot + M2 ln 4."""
    return M2_ln_c + M2 * np.log(4)


Masses= {"H2":2*constants.mH,
         "He":4*constants.mH,
         "Li":6*constants.mH,
         "N2":28*constants.mH,
         "O2":32*constants.mH,
         "Kr":83*constants.mH}

M2s={"H2":1.5487,
     "He":0.7525,
     "O2":4.652,
     "N2":4.742,
     "Li":6.037,
     "Kr":7.88}

Cs={"H2":2.4036,
     "He":C_from_Mlnc(0.7525, -0.3800),
     "O2":4.9969,
     "N2":4.8612,
     "Li":C_from_Mlnc(6.037, 12.71),
     "Kr":C_from_Mlnc(7.34, -2.62)}

Colors={"H2":'DarkRed',
        "He":'DarkOrange',
        "Li":'dimgrey',
        "N2":'DarkBlue',
        "O2":'DarkGreen',
        "Kr":'purple'}

#Nb: still need oscillator strengths for Li
allgases=['H2','He','N2','O2','Kr']

def GetInelasticCrossSection(gas):
    return opticalapprox.GetBetheBornCrossSectionFunction(M2s[gas],Cs[gas])

def GetKinematicFunctionInelastic(gas):
    return opticalapprox.GetKinematicFunction("./Data/OscillatorStrengths/OscStrength"+gas+".txt",Masses[gas])

