import constants
import etcross
import kinematics
import numpy as np
import pickle



def GetNextCollision(CrossSections,KinFunctions,T=20e3,Density=1e20,rng=None):

    rng = rng or np.random.default_rng()

    ListOfXs={}
    for k in CrossSections.keys():
        ListOfXs[k] = CrossSections[k](T)

    TotalXS       = sum(ListOfXs.values())
    MeanFreePath  = 1/(TotalXS*Density)

    Distance      = rng.exponential(scale=MeanFreePath, size=1)[0]

    CollisionType = rng.choice(len(ListOfXs.keys()), size=1, p=np.array(list(ListOfXs.values())) / TotalXS)
    CollisionTypeName=np.array(list(ListOfXs.keys()))[CollisionType][0]

    SplitName=CollisionTypeName.split("_")
    if(len(SplitName)==2):
        print(SplitName[0],SplitName[-1])
        dT, Theta = KinFunctions[SplitName[0]](T,int(SplitName[-1]))
    else:
        dT, Theta = KinFunctions[SplitName[0]](T)


    return Distance, Theta, dT, CollisionTypeName
