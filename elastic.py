import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

dat_Edep_a=pd.read_csv("./Data/Elastic/atoms_sigma_vs_energy_1-20keV.csv",skiprows=3)
dat_Edep_m=pd.read_csv("./Data/Elastic/molecules_sigma_vs_energy_1-20keV.csv",skiprows=6)

ElasticTotalXS = {}

for c in dat_Edep_a.columns:
    if ("_el" in c) and (not ("Abs") in c):
        label = c.split("_")[0]
        ElasticTotalXS[c] = interp1d(dat_Edep_a.E_keV, dat_Edep_a[c] * 1e-4)

for c in dat_Edep_m.columns:
    if ("el_corrected" in c) and (not ("Abs") in c):
        label = c.split("_")[0]
        ElasticTotalXS[c] = interp1d(dat_Edep_m.E_keV, dat_Edep_m[c] * 1e-4)

from scipy.interpolate import LinearNDInterpolator

def GetElasticCrossSectionInterpolator(filename="./Data/Elastic/AngleTables/Kr_DCS_1-20keV.csv"):
    f=open(filename,'r')
    lines=f.readlines()
    f.close()
    ci=0
    for line in lines:
        if("theta_deg" in line):
            break
        ci=ci+1
    dat_Kr_angle=pd.read_csv(filename,skiprows=ci)
    PointsToInterp=[]
    ValsToInterp=[]
    for c in dat_Kr_angle.columns:
        if "DCS" in c:
            label=float(c.split("_")[1][0:-3])

        for i in range(0,len(dat_Kr_angle)):
            PointsToInterp.append([label,dat_Kr_angle.theta_deg.iloc[i]])
            ValsToInterp.append(dat_Kr_angle[c].iloc[i]*2*np.pi*np.sin(dat_Kr_angle.iloc[i].theta_deg*np.pi/180))
    DiffXSElastic  = LinearNDInterpolator(PointsToInterp,ValsToInterp)
    return DiffXSElastic

def ElasticSampleFunction(T,filename="./Data/Elastic/AngleTables/Kr_DCS_1-20keV.csv",points=1000):
    DiffXSElastic=GetElasticCrossSectionInterpolator(filename)
    thetas=np.linspace(0,180,points)
    Slice=DiffXSElastic(T,thetas)
    return interp1d(np.cumsum(Slice)/sum(Slice),thetas)

def GetElasticKinematicFunction(filename="./Data/Elastic/AngleTables/Kr_DCS_1-20keV.csv"):
    def KinematicFunction(T):
        dTh=ElasticSampleFunction(T,filename)(np.random.uniform())
        dT=0
        return dT,dTh
    return KinematicFunction
