import matplotlib.pyplot as plt # Package for plotting
import numpy as np # Package for scientific computing

from Utils.beams import * # Functions for mesh generation

from Utils.ESO_utils import * # Fucntions for FEM analysis and postprocessing
from Utils.BESO_utils import * # Fucntions for FEM analysis and postprocessing
from Utils.SIMP_utils import *

import solidspy.postprocesor as pos # SolidsPy package for postprocessing


# %% ESO stress based

def ESO_stress(length, height, nx, ny, dirs, positions, niter, RR, ER, V_opt, plot=False):
    length = 20
    height = 10
    nx = 50
    ny= 20
    dirs = np.array([[0,-1]])
    positions = np.array([[10,1]])
    elsI = np.copy(els)

    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

    IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    niter = 200
    RR = 0.001 # Initial removal ratio
    ER = 0.005 # Removal ratio increment
    V_opt = volume(els, length, height, nx, ny) * 0.50 # Optimal volume

    ELS = None
    for _ in range(niter):

        # Check equilibrium
        if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: break  # Check equilibrium/volume and stop if not
        ELS = els
        
        # FEW analysis
        IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
        UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Displacements, strains and stresses
        E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements
        vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)

        # Remove/add elements
        RR_el = vons/vons.max() # Relative stress
        mask_del = RR_el < RR # Mask for elements to be deleted
        mask_els = protect_els(els, loads, BC) # Mask for elements to be protected
        mask_del *= mask_els  
        els = np.delete(els, mask_del, 0) # Delete elements
        del_node(nodes, els) # Delete nodes that are not connected to any element

        RR += ER

    if plot:
        pos.fields_plot(elsI, nodes, UCI, E_nodes=E_nodesI, S_nodes=S_nodesI) # Plot initial mesh
        pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes) # Plot optimized mesh

        fill_plot = np.ones(E_nodes.shape[0])
        plt.figure()
        tri = pos.mesh2tri(nodes, ELS)
        plt.tricontourf(tri, fill_plot, cmap='binary')
        plt.axis("image");


# %% Eso stiff based

def optimization_ESO_stiff():
    length = 60
    height = 60
    nx = 60
    ny= 40
    dirs = np.array([[0,-1]])
    positions = np.array([[10,1]])
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

    IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    niter = 200
    RR = 0.005 # Initial removal ratio
    ER = 0.05 # Removal ratio increment
    V_opt = volume(els, length, height, nx, ny) * 0.20 # Optimal volume
    ELS = None
    for _ in range(niter):
        # Check equilibrium
        if not is_equilibrium(nodes, mats, els, loads) or volume(els, length, height, nx, ny) < V_opt: 
            print('gla')
            break # Check equilibrium/volume and stop if not
        
        # FEW analysis
        IBC, UG = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
        UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

        # Compute Sensitivity number
        sensi_number = sensi_el(nodes, mats, els, UC) # Sensitivity number
        mask_del = sensi_number < RR # Mask of elements to be removed
        mask_els = protect_els(els, loads, BC) # Mask of elements to do not remove
        mask_del *= mask_els # Mask of elements to be removed and not protected
        ELS = els # Save last iteration elements
        
        # Remove/add elements
        els = np.delete(els, mask_del, 0) # Remove elements
        del_node(nodes, els) # Remove nodes

        RR += ER


# %% BESO

def optimization_BESO():
    length = 20
    height = 10
    nx = 50
    ny= 20
    dirs = np.array([[0,-1]])
    positions = np.array([[21,10]])
    nodes, mats, els, loads, BC = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

    IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    niter = 200
    ER = 0.005 # Removal ratio increment
    t = 0.0001 # Threshold for error

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 1 # Radius for the sensitivity filter
    adj_nodes = adjacency_nodes(nodes, els) # Adjacency nodes
    centers = center_els(nodes, els) # Centers of elements

    Vi = volume(els, length, height, nx, ny) # Initial volume
    V_opt = Vi.sum() * 0.50 # Optimal volume

    # Initialize variables.
    ELS = None
    mask = np.ones(els.shape[0], dtype=bool) # Mask of elements to be removed
    sensi_I = None  
    C_h = np.zeros(niter) # History of compliance
    error = 1000 

    for i in range(niter):
        # Calculate the optimal design array elements
        els_del = els[mask].copy() # Elements to be removed
        V = Vi[mask].sum() # Volume of the structure

        # Check equilibrium
        if not is_equilibrium(nodes, mats, els_del, loads):  
            print('Is not equilibrium')
            break # Stop the program if the structure is not in equilibrium

        # Storage the solution
        ELS = els_del 

        # FEW analysis
        IBC, UG, rhs_vec = preprocessing(nodes, mats, els_del, loads) # Calculate boundary conditions and global stiffness matrix
        UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els_del, IBC, UG) # Calculate displacements, strains and stresses

        # Sensitivity filter
        sensi_e = sensitivity_els(nodes, mats, els, mask, UC) # Calculate the sensitivity of the elements
        sensi_nodes = sensitivity_nodes(nodes, adj_nodes, centers, sensi_e) # Calculate the sensitivity of the nodes
        sensi_number = sensitivity_filter(nodes, centers, sensi_nodes, r_min) # Perform the sensitivity filter

        # Average the sensitivity numbers to the historical information 
        if i > 0: 
            sensi_number = (sensi_number + sensi_I)/2 # Average the sensitivity numbers to the historical information
        sensi_number = sensi_number/sensi_number.max() # Normalize the sensitivity numbers

        # Check if the optimal volume is reached and calculate the next volume
        V_r = False
        if V <= V_opt:
            els_k = els_del.shape[0]
            V_r = True
            break
        else:
            V_k = V * (1 + ER) if V < V_opt else V * (1 - ER)

        # Remove/add threshold
        sensi_sort = np.sort(sensi_number)[::-1] # Sort the sensitivity numbers
        els_k = els_del.shape[0]*V_k/V # Number of elements to be removed
        alpha_del = sensi_sort[int(els_k)] # Threshold for removing elements

        # Remove/add elements
        mask = sensi_number > alpha_del # Mask of elements to be removed
        mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, BC) # Mask of elements to be protected
        mask = np.bitwise_or(mask, mask_els) 
        del_node(nodes, els[mask], loads, BC) # Delete nodes

        # Calculate the strain energy and storage it 
        C = 0.5*rhs_vec.T@UG
        C_h[i] = C
        if i > 10: error = C_h[i-5:].sum() - C_h[i-10:-5].sum()/C_h[i-5:].sum()

        # Check for convergence
        if error <= t and V_r == True:
            print("convergence")
            break

        # Save the sensitvity number for the next iteration
        sensi_I = sensi_number.copy()

# %% SIMP

def optimization_SIMP(n_elem, volfrac):
    # Initialize variables
    length = 60
    height = 60
    nx = n_elem
    ny= n_elem
    niter = 60
    penal = 3 # Penalization factor
    Emin=1e-9 # Minimum young modulus of the material
    Emax=1.0 # Maximum young modulus of the material

    dirs = np.array([[0,-1], [0,1], [1,0]])
    positions = np.array([[61,30], [1,30], [30, 1]])
    nodes, mats, els, loads = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions, n=1)

    # Initialize the design variables
    change = 10 # Change in the design variable
    g = 0 # Constraint
    rho = volfrac * np.ones(ny*nx, dtype=float) # Initialize the density
    sensi_rho = np.ones(ny*nx) # Initialize the sensitivity
    rho_old = rho.copy() # Initialize the density history
    d_c = np.ones(ny*nx) # Initialize the design change

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
    centers = center_els(nodes, els) # Calculate centers
    E = mats[0,0] # Young modulus
    nu = mats[0,1] # Poisson ratio
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8]) # Coefficients
    kloc = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], 
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]); # Local stiffness matrix
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

    iter = 0
    for _ in range(niter):
        iter += 1

        # Check convergence
        if change < 0.01:
            print('Convergence reached')
            break

        # Change density 
        mats[:,2] = Emin+rho**penal*(Emax-Emin)

        # System assembly
        stiff_mat = sparse_assem(els, mats, neq, assem_op, kloc)
        rhs_vec = ass.loadasem(loads, bc_array, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(bc_array, nodes, disp)

        compliance = rhs_vec.T.dot(disp)

        # Sensitivity analysis
        sensi_rho[:] = (np.dot(UC[els[:,-4:]].reshape(nx*ny,8),kloc) * UC[els[:,-4:]].reshape(nx*ny,8) ).sum(1)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)

        # Optimality criteria
        rho_old[:] = rho
        rho[:], g = optimality_criteria(nx, ny, rho, d_c, g)

        # Compute the change
        change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)

    plt.ion() 
    fig,ax = plt.subplots()
    ax.imshow(-rho.reshape(n_elem,n_elem), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    ax.set_title('Predicted')
    fig.show()