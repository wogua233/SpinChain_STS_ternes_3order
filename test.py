from draw_spec import *
from fit_graph import *
from draw_spec_parallel import gui_drawspec_parallel

natoms=4 # nur of atom
T=8 #K
T=T* 0.08617
j12=1.5 # heisenberg_matrix
j14=1.5 # heisenberg_matrix
j23=1.5 # heisenberg_matrix
j34=1.5 # heisenberg_matrix
# j45=1.5 # heisenberg_matrix
# j56=1.5 # heisenberg_matrix
A=2.6e-11 # tip-sample interaction
J=-0.6 # KONDO coupling

# ______________________________________________________________
atom=Atom()
atom.natoms=natoms
atom.J=J
experiment = Experiment()
experiment.atom=atom
experiment.init_matrix()
experiment.T=T
experiment.A=A
experiment.matrix[0, 0] =j12
experiment.matrix[0, 2] =j14
experiment.matrix[1, 1] =j23
experiment.matrix[2, 2] =j34
# experiment.matrix[3, 3] =j45
# experiment.matrix[4, 4] =j56
calcStore=CalcStore()
gui_drawspec(experiment,calcStore,debug=False)


plot_pairs_2n(natoms=experiment.atom.natoms,calc=calcStore,use_dialog=True)
