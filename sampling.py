import constants
import etcross
import kinematics
import numpy as np
import pickle



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



# Tool to generate next collision based on provided cross sections and
#   kinematic functions.

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
