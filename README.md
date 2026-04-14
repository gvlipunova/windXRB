<b>Calculate N_H along LOS to NS over the period of a binary.</b> 

The binary contains optical star with a spherical wind and NS with  accretion wake where density is increased by some factor compared to the wind.  
The wake side at the current moment coincides with the locus of the wind particles that "met" NS in the past.

<img width="456" height="412" alt="image" src="https://github.com/user-attachments/assets/d07495eb-df0e-4958-bd98-925b50788d10" />

The extension of the wake is set by the fraction of the NS orbit, "remembered" by the wake.
The width of the wake equals the accretion radius (variable along the orbit) times user-defined factor. 
The head of the wake is missing so far.

User <b>parameters</b> should be listed in XRB.ini. See default parameters at the bottom (in the future). XRB_Blondin.ini contains parameters for a binary from paper [Blondin et al (1990)](url)

See <b>results for GX 301−2</b> in folder GX_dif_geo/.

--------------------------------
<b>How to run</b>

1. to plot ionization map, run: 

<tt>mpiexec -n 8 python xi_map_mpi.py [phase]</tt>

where phase is a float number >=0 and <=1


<img width="508" height="450" alt="image" src="https://github.com/user-attachments/assets/86590401-861a-4df3-8229-b261175be56e" />


2. to calculate NH(phase) for the whole period, run:

<tt>mpiexec -n 8 python NH_period_inclined_mpi.py</tt>

It will produce PDF files and ASCII *dat files in folder data/ 


3. to plot different models NH(phase), run:

<tt>python plot_many_NH_period.py [dirname] [parameter names]</tt>

where  
  - dirname is the path to the folder with *dat files made beforehand by NH_period_inclined_mpi

  - parameter names is optional argument: parameters separated by spaces to indicate in the legends. Name of parameters may be found in the headers of dat-files.

<img width="645" height="383" alt="image" src="https://github.com/user-attachments/assets/3389118d-d18d-4d19-845d-3a710d2aa74e" />
