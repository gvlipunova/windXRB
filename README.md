Draft of instructions:
1. to plot ionization map run: 

mpiexec -n 8 python xi_map_mpi.py [phase]

where phase is a float number >=0 and <=1


2. to calculate NH(phase) for the whole period run:

mpiexec -n 8 python NH_period_inclined_mpi.py

It will produce PDF files and ASCII *dat files in folder data/ 


3. to plot different models NH(phase) run:

python plot_many_NH_period.py [dirname]

where dirname is the path to the forlder with dat files made beforehand by NH_period_inclined_mpi
