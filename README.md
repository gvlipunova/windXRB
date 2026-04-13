Draft of instructions:

User parameters should be listed in XRB.ini.

XRB_Blondin.ini contains parameters for a binary from paper [Blondin et al (1990)](url) 

1. to plot ionization map, run: 

mpiexec -n 8 python xi_map_mpi.py [phase]

where phase is a float number >=0 and <=1


<img width="508" height="450" alt="image" src="https://github.com/user-attachments/assets/86590401-861a-4df3-8229-b261175be56e" />


2. to calculate NH(phase) for the whole period, run:

mpiexec -n 8 python NH_period_inclined_mpi.py

It will produce PDF files and ASCII *dat files in folder data/ 


3. to plot different models NH(phase), run:

python plot_many_NH_period.py [dirname]

where dirname is the path to the folder with *dat files made beforehand by NH_period_inclined_mpi

<img width="645" height="383" alt="image" src="https://github.com/user-attachments/assets/3389118d-d18d-4d19-845d-3a710d2aa74e" />
