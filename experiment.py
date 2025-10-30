from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

from dataclasses import dataclass
from typing import Union, Tuple

Number = Union[int, float]
Vec3 = Tuple[Number, Number, Number]

@dataclass
class Atom:
    """
    Minimal atom container compatible with your spin-transport code.
    Notes:
      - J can be a scalar (isotropic) or a 3-tuple (Jx, Jy, Jz).
      - S is the spin quantum number.
    """
    S: float = 0.5     # spin
    g: float = 2.0     # g-factor
    D: float = 0.0     # axial anisotropy
    E: float = 0.0     # transverse anisotropy
    J: Union[Number, Vec3] = -0.04  # exchange coupling (scalar or 3-tuple) kondo scattering
    U: float = 0.0     # potential scattering strength
    w: float = 20.0    # bandwidth / cutoff
    natoms: int =2




ArrayLike = Union[List[float], np.ndarray]

@dataclass
class Experiment:
    # Core physics knobs
    T: float = 1 * 0.08617                 # temperature (meV)
    xrange: float = 10.0                    # voltage span (mV) or grid elsewhere
    lt: float = 0.005                       # lifetime broadening (meV)

    # Tip / sample spin polarizations (Bloch vectors)
    ptip: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    psample: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))

    # Spin ensemble / atoms
    atom=Atom()
    atom: Optional[object] = atom          # you can pass in your 'atomlink' object

    # Eigen system (filled later by your solver)
    Eigenvec=None
    Eigenval= None

    # Geometry / positions (1-based in original Scilab code)
    position: int = 1
    jposition: int = 1

    # Plot / scaling parameters
    A: float = 1.0 # tip-sample interaction
    b: float = 0.0
    x0: float = 0.0
    y0: float = 0.0

    # Magnetic field (Tesla or same unit as your Hamiltonian expects)
    B: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 4.0], dtype=float))

    # Extra configuration strings
    # 先创建 4×4 复数型零矩阵
    matrix = np.zeros((atom.natoms-1, atom.natoms-1), dtype=float)
    matrixDM = None


    # 按行/列手动填入矩阵元（示例：全 0 矩阵，您替换成自己的数值）
    matrix[0, 0] = 1.0  # j12
    # matrix[0, 3] = 2.0  # j15
    # matrix[1, 1] = 3.0  # j23
    # matrix[2, 2] = 4.0  # j34
    # matrix[3, 3] = 5.0  # j45
    # print(mat)

    # Couplings / flags
    heisenberg_coupling: float = 1.0 # C
    sample_entanglement: bool = True
    sef: float = 1.0

    # Paramagnet options
    paramagnetic: bool = False
    paramag_S: float = 5.0 / 2.0
    paramag_g: float = 2.0

    # Misc numeric knobs
    eta: float = 0.3
    no_eval: int = 1000
    max_no_eigenstates: int = 50

    # Calculation switches
    third_order_calc: bool = True
    rate_calc: bool = False
    entanglement: bool = False
    allatomsequ: int = 0

    # Optional explicit voltage grid (overrides xrange when provided)
    xgrid: Optional[np.ndarray] = None

    def init_matrix(self):
        print('原子数：',self.atom.natoms)
        self.matrix = np.zeros((self.atom.natoms - 1, self.atom.natoms - 1), dtype=float)
    def set_tip_polarization(self, v: ArrayLike):
        self.ptip = np.asarray(v, dtype=float)

    def set_sample_polarization(self, v: ArrayLike):
        self.psample = np.asarray(v, dtype=float)

    def set_field(self, v: ArrayLike):
        self.B = np.asarray(v, dtype=float)

    def set_eigensystem(self, eigenvec: np.ndarray, eigenval: np.ndarray):
        self.Eigenvec = np.asarray(eigenvec)
        self.Eigenval = np.asarray(eigenval)

#____________________________#
if __name__ == "__main__":
    # experiment = Experiment()
    # print(experiment.atom.w)
    atom=Atom(natoms=1)
    print(atom.natoms)
