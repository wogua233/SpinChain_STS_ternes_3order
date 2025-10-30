from draw_spec import *
from fit_graph import *
from draw_spec_parallel import gui_drawspec_parallel

if __name__ == '__main__':
    xrange=20
    natoms=6 # nur of atom
    T=8.5 #K
    T=T* 0.08617
    j12=1.4 # heisenberg_matrix
    j16=1.2 # heisenberg_matrix
    j23=1.8 # heisenberg_matrix
    j34=1.4 # heisenberg_matrix
    j45=1.6 # heisenberg_matrix
    j56=1.7 # heisenberg_matrix
    jmat=[j12,j16,j23,j34,j45,j56] # 作图用
    A=2.6e-11 # tip-sample interaction
    J=-0.6 # KONDO coupling

    save_calc=True # 是否保留计算的谱线
    # 计算下面部分用时
    import time
    start_time = time.time()
    # ______________________________________________________________
    atom=Atom()
    atom.natoms=natoms
    atom.J=J
    experiment = Experiment()
    experiment.atom=atom
    experiment.xrange=xrange
    experiment.init_matrix()
    experiment.T=T
    experiment.A=A
    experiment.matrix[0, 0] =j12
    experiment.matrix[0, 4] =j16
    experiment.matrix[1, 1] =j23
    experiment.matrix[2, 2] =j34
    experiment.matrix[3, 3] =j45
    experiment.matrix[4, 4] =j56
    calcStore=CalcStore()
    gui_drawspec_parallel(experiment,calcStore,debug=False,savedata=False)
    # 用时
    end_time = time.time()
    print(f"计算时间: {end_time - start_time} 秒")

    plot_pairs_2n(natoms=experiment.atom.natoms,calc=calcStore,use_dialog=True,matrix=jmat)


# ______________________________________________________________
    if save_calc:
        for i in range(natoms):
            np.savetxt(f"didv_atom{i+1}.dat",calcStore.spec)