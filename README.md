# Spectral-function-of-a-phonon-unfolding


PAPER:
Zacharias, Marios, George Volonakis, Feliciano Giustino, and Jacky Even. "Anharmonic electron-phonon coupling in ultrasoft and locally disordered perovskites." arXiv preprint arXiv:2302.09625 (2023).


Code for Spectral function of a phonon unfolding problem
For systems undergoing static symmetry breaking due to lattice distortion coming, e.g., from defects, atomic disorder, or a charge density wave, a supercell is required to compute the phonons. In this case, the crystal's symmetry operations (translations and rotations) are no longer applicable and all atoms in the supercell need to be displaced for calculating the dynamical matrix and, hence, the renormalized phonon frequencies ω_Qμ, where Q and μ are the phonon wavevector and band indices. To illustrate the effect of lattice distortion in the phonons, a common practise is to employ phonon unfolding and evaluate the momentum-resolved spectral function given by  ^73 :
A_q (ω)=∑_Qμ▒  P_(Qμ,q) δ(ω-ω_Qμ )
Here q denotes a wavevector in the Brillouin zone of the unit cell and P_(Qμ,q) represents the spectral weights which are evaluated in the spectral representation of the singleparticle Green's function as  ^74 :
P_(Qμ,q)=1/N_g   Ω/Ω ˜  ∑_αj▒  |∑_κ▒  e ˜_(ακ,μ) (q)e^(i(q+g_j )⋅τ ‾_κ ) |^2,
where j is an index for the reciprocal lattice vectors g of the unit's cell Brillouin zone, α denotes a Cartesian direction, and κ is the atom index. The symbol ∼ indicates quantities calculated using the disordered structure. N_g acts as a normalization factor representing the total number of reciprocal lattice vectors entering the summation. The spectral weight can be understood, essentially, as the projection of the phonon eigenvector e ˜_(ακ,μ) (Q) on the phonon eigenvectors e_(ακ,ν) (q) computed in the unit cell, given that Q unfolds into q via Q=q+g_j-G, where G is a reciprocal lattice vector of the distorted structure.
