import constants
import etcross
import kinematics
import numpy as np
import pickle

rng_seed=12345
rng = np.random.default_rng(rng_seed)


f=open("TotalCrossSections.pkl",'rb')
CrossSections=pickle.load(f)
f.close()

def GetNextCollision(T=20e3,Density=1e20):
    ListOfXs={}
    for k in CrossSections.keys():
        ListOfXs[k] = CrossSections[k](T)

    TotalXS       = sum(ListOfXs.values())
    MeanFreePath  = 1/(TotalXS*Density)

    Distance      = rng.exponential(scale=MeanFreePath, size=1)[0]

    CollisionType = rng.choice(len(ListOfXs.keys()), size=1, p=np.array(list(ListOfXs.values())) / TotalXS)
    CollisionTypeName=np.array(list(ListOfXs.keys()))[CollisionType][0]

    # Make single diff sample functions for elastic and discrete
    SampleDictInel=etcross.MakeSampleDicitonary(T,constants.movM,ns=range(2,8))
    SampleFuncEl=etcross.MakeSampleFunction_elastic(T,constants.movM)

    # Set up grid for sampling double diff continuum cross section
    #  It will sample from a 2D histogram with E/R between Emin and
    #  Emax, logKa2min and logka2max, with dimensionality of bins
    #  in each direction.  Use the example in TechNotePlots to make
    #  sure this suitably spans the space where the cross section is
    #  significant, if in doubt.
    Emin=1.00001;  Emax=10; logKa2min=-9; logKa2max=6; bins=100
    EE,KK = np.meshgrid(np.linspace(Emin,Emax,bins),np.linspace(logKa2min,logKa2max,bins))
    Distn_continuum = etcross.DistributionDraw(EE,KK,T,constants.movM)

    if(CollisionTypeName=='elastic'):
        Ka2   = etcross.GetRandomKa2_elastic(SampleFuncEl,1)[0]
        dT    = kinematics.getDeltaT(Ka2,0)
        Theta = np.arccos(kinematics.getCosTheta(Ka2,T,0))
    elif(CollisionTypeName=='continuum'):
        Theta,dT,E,logK=etcross.SampleContinuum(T,Distn_continuum,Emin,Emax,logKa2min,logKa2max,samples=1)
        Theta = Theta[0]
        dT   =  dT[0]
    elif('discrete' in CollisionTypeName):
        n     = int(CollisionTypeName.split("_")[1])
        dE    = kinematics.En(n)
        Ka2   = etcross.GetRandomKa2_discrete(n,SampleDictInel,1)[0]
        dT    = kinematics.getDeltaT(Ka2,dE)
        Theta = np.arccos(kinematics.getCosTheta(Ka2,T,dE))


    return Distance, Theta, dT, CollisionTypeName
