import numpy as np

T_e       = 18e3                       # Default electron energy [eV]
R         = 13.6                       # Rydberg constant        [ev]
BohrRad   = 5.3e-11                    # Bohr radius             [m]
me        = 511e3                      # Electron mass           [eV]
mT        = 3e9                        # Tritium mass            [eV]
hbarc     = 197e-9                     # hbar*c for unit conversion [eV m]
movM=(me+mT)/mT                        # me / mu                 [dimensionless]
NormConst = 4*np.pi*BohrRad**2         # Cross section prefactor [m^2]