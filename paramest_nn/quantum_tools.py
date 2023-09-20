"""
Quantum Tools Module

This module provides a collection of functions and a QuantumModel class for performing simulations and operations on quantum systems.
It includes various tools for generating quantum models, simulating quantum dynamics, computing probabilities, and more.

Author: Carlos Sánchez Muñoz
Date: 29/08/2023
GitHub Repository: 

Classes:
    QuantumModel: A class representing a quantum model with open-system dynamics.

Functions:
    - create_TLS_model: Create a QuantumModel instance for a two-level system (TLS) with the given parameters.
    - generate_clicks_TLS: Simulates a quantum trajectory for a coherently driven two-level-system (TLS), generating a fixed number of time delays between quantum jumps.
    - to_time_delay: Convert an array of values into an array of time delays between consecutive values.
    - to_time_delay_matrix: Convert a matrix of values into time delays between consecutive values.
    - generate_prob_from_model_list: Returns a normalized probability distribution P(D|model) for a list of models. 
    - compute_log_likelihood_analytical: Compute the logarithm of the likelihood analytically for the TLS model.
    - compute_neg_log_likelihood_analytical: Compute the negative logarithm of the likelihood analytically for the TLS model.
    - compute_likelihood_analytical: Compute the likelihood analytically for the TLS model.
    - get_estimates_Bayesian: Calculate estimates of Delta (1D parameter estimation) in a coherently driven TLS by computing the probability distribution over a grid of Delta values.
    - prob_no_click: Calculates the probability of no jump occurring within a time delay tau after the last jump in a coherently driven TLS.
    - compute_waiting_time_list: Calculate the waiting time distribution of a coherently driven two-level system for a list of time delays.
    - compute_waiting_time: Calculates the waiting time distribution of a coherently driven two-level system based on Omega and Delta.
    - compute_population_ss: Calculate the steady-state population of a coherently driven two-level system based on Omega and Delta.
    - compute_likelihood: Computes (using Qutip) the likelihood function L(D|Delta) given the model parameter Delta.
    - compute_negative_log_likelihood: Returns the negative log-likelihood function -log[L(D|Delta)].
    - create_directory: Looks for a directory, creates it if it is not found.

"""

from qutip import *
import numpy as np
from scipy import optimize

from scipy import stats
from scipy.linalg import expm
from scipy.stats import norm
from time import time
import os

from numpy import exp, cos, cosh, sqrt

class QuantumModel:
    """
    A class representing a quantum model with open-system dynamics.

    Attributes:
    -----------
        H (qobj): The system Hamiltonian.
        c_ops (list of qobj): List of collapse operators representing system-environment interactions.
        J (array): Jump matrix corresponding to collapse operators.
        L (qobj): Liouvillian operator representing the full dynamics.
        psi0 (qobj): Initial state of the quantum system.
        Lhat (qobj): Modified Liouvillian by subtracting the Jump matrix.
        rho0 (array): Vectorized initial density matrix.
        Ehat (array or None): Eigenvalue matrix of the modified Liouvillian (Lhat).
        Einvhat (array or None): Inverse of the Eigenvalue matrix (Ehat) if available.
        eigs_hat (array or None): Eigenvalues of the modified Liouvillian (Lhat).

    Methods:
    -----------
        simulateTrajectories(self, tfin=float, ntraj=int, tlistexp=None, seed=None):
            Generates an array of Monte Carlo jumps from the given quantum model.

        simulateTrajectoryFixedJumps(self, njumpsMC=48):
            Simulate a single quantum trajectory with a fixed total number of quantum jumps.

        computeLikelihood(self, data: np.array, r: float = 5.0, tfin=-1, method="spectral") -> np.array:
            Returns the likelihood function L(D|model) proportional to the probability of the data D being measured,
            given a particular model (Hamiltonian + quantum jump operators).
    """

    def __init__(self,H,c_ops,J,psi0,with_eigvals=True):
        """
        Initialize the QuantumModel class.

        Parameters:
        -----------
            H (qobj): System Hamiltonian.
            c_ops (list of qobj): List of collapse operators.
            J (array): Jump matrix.
            psi0 (qobj): Initial state of the quantum system.
            with_eigvals (bool): Whether to calculate eigenvalues and eigenvectors of the modified Liouvillian.

        """
        self.H = H
        self.c_ops = c_ops
        self.J = J.full()
        self.L = liouvillian(H,c_ops)
        self.psi0 = psi0
        self.Lhat = self.L-J
        self.rho0 = operator_to_vector(ket2dm(psi0)).full()
        
        if with_eigvals==True:
            eigen_hat=self.Lhat.eigenstates()
            h2 = len(eigen_hat[1])
            E = np.transpose(np.reshape(
                np.array([eigen_hat[1][i].full() for i in range(h2)]), 
                (h2,h2)))
            Einv = np.linalg.inv(E)
            self.Ehat = E
            self.Einvhat = Einv
            self.eigs_hat = eigen_hat[0]
        else:
            self.Ehat = None
            self.Einvhat = None
            self.eigs_hat = None

    def simulateTrajectories(self, tfin: float, ntraj:int, tlistexp = None, seed = None, progress_bar = False, num_cpus = None):
        """
        Generates an array of Monte Carlo jumps from the given quantum model.

        This function simulates quantum jumps for the provided model using the Monte Carlo method with Qutip.

        Parameters:
        -----------
            tfin (float): Final time of the simulation.
            ntraj (int): Number of trajectories to simulate.
            tlistexp (array or None): If provided, an array of times where expectation values should be computed.
            seed (int or None): Seed value for random number generator.
            progress_bar (bool): Whether to display a progress bar in the calculation.

        Returns:
        -----------
            tuple or list: Depending on the presence of tlistexp.
                If tlistexp is None:
                    List of arrays containing time delays between jumps for each trajectory.
                If tlistexp is not None:
                    A tuple containing:
                        - List of arrays containing time delays between jumps for each trajectory.
                        - Array of corresponding to the expectation values of a quantum operator.

            """

        H = self.H
        c_ops = self.c_ops
        psi0 = self.psi0

        options = Options()
        if seed != None:
            options.seeds = np.arange(seed,seed+ ntraj)

        if num_cpus is not None:
            options.num_cpus = num_cpus

        if tlistexp is None:
            options.store_final_state = False
            tlist = [0,tfin]
            sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=ntraj,progress_bar=progress_bar,options=options)
            times = sol.col_times
            taus = []
            for time_jumps in times:
                if len(time_jumps)>0:
                    taus.append(to_time_delay(time_jumps))
                else:
                    taus.append(np.array([]))
            return taus
        else:
            options.store_states = True
            tlist = tlistexp
            sigma = c_ops[0]
            sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=ntraj,progress_bar=progress_bar,options=options)
            times = sol.col_times
            taus = []
            for time_jumps in times:
                if len(time_jumps)>0:
                    taus.append(to_time_delay(time_jumps))
                else:
                    taus.append(np.array([]))
            statestraj = sol.states
            pop_array = np.asarray([[expect(sigma.dag()*sigma, state) for state in statetime] for statetime in statestraj])
            return taus, pop_array


    def simulateTrajectoryFixedJumps(self,njumpsMC=48):
        """
        Simulate a single quantum trajectory with a fixed total number of quantum jumps.

        This function simulates a quantum jump process using the Monte Carlo method to generate a set of time delays.
        The number of time delays generated is controlled by njumpsMC. Only one trajectory is simulared.

        Parameters:
        -----------
        njumpsMC (int): The desired number of time delays to be generated.

        Returns:
        -----------
        np.array: Array of time delays representing quantum jumps.

        """
        
        H = self.H
        c_ops = self.c_ops
        psi0 = self.psi0
            
        t0 = 0
        factor = 1.2

        jumpOp = c_ops[0]
        rho_ss = steadystate(H,c_ops)
        n_expect = expect(jumpOp.dag()*jumpOp, rho_ss)

        tf=factor*njumpsMC/n_expect
        tlist=[t0,tf] 

        options = Options()
        options.store_final_state = True
        
        # This flag will mark when we reach the minimum number of jumps required by njumpsMC
        flag = 0
        taus_list = []

        tshift = t0
        while flag ==0:
                
                sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=1,progress_bar=False,options=options)
            
                # We prepare the inputs of the next simulation in case the total number of jumps did not reach the minimum required of njumpsMC
                psi0 = sol.states[-1,0]
                times = tshift + sol.col_times[0]
                tshift = times[-1]
                taus = list(to_time_delay(times))

                # Append the results to taus_list
                taus_list += taus

                # If the minimum number of jumps was reached, we raise our flag to stop the simulation
                if len(taus_list)>= njumpsMC:
                    flag = 1

        taus = np.array(taus_list)
        return taus[:njumpsMC]
    

    def computeLikelihood(self,data: np.array, r: float = 5.0,tfin=-1,method="spectral") -> np.array:
        """
        Returns the likelihood function L(D|model) proportional to the probability of the data D being measured, given a particular model (Hamiltonian + quantum jump operators).
        This calculation is based on purely numerical calculations using Qutip. 
        This calculation does not assume any particular system, and takes an instance of the 'model' class as an input.

        Parameters:
        -----------
        data (np.array): The trajectory data in the form of an array of time delays
        r (float): Renormalization factor applied to avoid small floats in the calculation of probabilities.
        model (QuantumModel):  Instance of QuantumModel class that contains the Hamiltonian and quantum jump operators that defines the model.
        tfin: Final time of the simulation. If -1, final time is set by the time when the last jump in the data array occurs (i.e., tfin = sum(data))
        method: Method used for the simulation. Can be "spectral" or "direct". If "spectral", a spectral decomposition of the Liouvillian is used.

        Returns:
        -----------
        np.array: Unnormalized likelihood function.
        """

        tau_list = data
        t_total = np.sum(tau_list)
        njumps = len(tau_list)
    
        J = self.J
        psi0=self.psi0
        Lhat = self.Lhat
    
        hilbert = psi0.shape[0]
        rho0 = self.rho0
        rhoC = rho0
        rhoC_t = []
        
        if method=='spectral':
            # Spectral decomposition of (L-J)
            E=self.Ehat 
            Einv=self.Einvhat
            EinvTr = np.transpose(Einv)
            eigvals = self.eigs_hat

                
            Uops = [  E@(np.transpose((exp(tau*eigvals))*EinvTr))       for tau in tau_list]
        
        
            # Now we need to use numpy arrays    
                
            for jump_idx  in range(njumps):
                renorm = tau_list[jump_idx]*r
                rhoC=J@Uops[jump_idx]@rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)

            # If the simulation set a final time (not a fixed # of clicks):
            if tfin!= -1:
                tau_fin = tfin-t_total
                renorm = tau_fin*r
                Ufin = E@(np.transpose((exp(tau_fin*eigvals))*EinvTr)) 
                rhoC=Ufin@rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)
            
            likelihoodTime = np.array([np.reshape(rhoCsel,(hilbert,hilbert)).trace() for rhoCsel in rhoC_t])   

        if method=='direct':
            Uops = [(tau*Lhat).expm() for tau in tau_list]

            for jump_idx  in range(njumps):
                renorm = tau_list[jump_idx]*r
                rhoC=J*Uops[jump_idx]*rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)

            # If the simulation set a final time (not a fixed # of clicks):
            if tfin!= -1:
                tau_fin = tfin-t_total
                renorm = tau_fin*r
                Ufin = (tau_fin*Lhat).expm()
                rhoC=Ufin*rhoC
                rhoC = renorm*rhoC
                rhoC_t.append(rhoC)

            likelihoodTime = np.array([vector_to_operator(rhoCsel).tr() for rhoCsel in rhoC_t])   

        return likelihoodTime

def create_TLS_model(delta,omega,gamma=1):
    """
    Create a QuantumModel instance for a two-level system (TLS) with the given parameters.

    Parameters:
    -----------
    delta (float): Energy difference between the two levels.
    omega (float): Rabi frequency of the drive.
    gamma (float, optional): Dissipation rate. Normally sets the time-energy scale, so default is 1.

    Returns:
    --------
    QuantumModel: An instance of the QuantumModel class representing the open two-level quantum system.

    """
    psi0=basis(2,0); a=destroy(2); c_ops=[]; c_ops.append(np.sqrt(gamma)*a)
    H=delta*a.dag()*a+omega*a+omega*a.dag()
    J = gamma*sprepost(a,a.dag())

    model_TLS = QuantumModel(H,c_ops,J,psi0)
    return model_TLS

def generate_clicks_TLS(params,njumpsMC=48,gamma=1.):
    """
    Simulates a quantum trajectory for a coherently driven two-level-system (TLS), generating a fixed number of time delays between quantum jumps.

    Parameters:
    -----------
    params (list): A list containing delta and omega parameters for the TLS model.
    njumpsMC (int, optional): The desired number of time delays to be generated. Default is 48.
    gamma (float, optional): Dissipation rate. Normally sets the time-energy scale, so default is 1.

    Returns:
    --------
    np.array: Array of time delays representing quantum jumps.

    """
    delta = params[0]
    omega = params[1]

    model = create_TLS_model(delta,omega,gamma)
    taus = model.simulateTrajectoryFixedJumps(njumpsMC)
    return taus

def to_time_delay(array: np.array):
    """
    Convert an array of values into an array of time delays between consecutive values.

    Parameters:
    -----------
        array (numpy.ndarray): Input array of values.

    Returns:
    -----------
        numpy.ndarray: Array of time delays between consecutive values.

    """
    return np.concatenate((np.asarray([array[0]]), np.diff(array)))

def to_time_delay_matrix(matrix: np.array):
    """
    Convert a matrix of values into a matrix of time delays between consecutive values.

    Parameters:
    -----------
        matrix (numpy.ndarray): Input matrix of values.

    Returns:
        numpy.ndarray: Matrix of time delays between consecutive values.

    """
    first_column = np.reshape(matrix[:, 0], (matrix.shape[0], 1))
    time_delay_matrix = np.concatenate((first_column, np.diff(matrix)), axis=1)
    return time_delay_matrix

def generate_prob_from_model_list(data: np.array, model_list : list, r: float = 5.0, tfin=-1, method = "spectral"):
    """
    Returns a normalized probability distribution P(D|model) over a list of models. 

    Parameters:
    -----------
    data (np.array): The trajectory data in the form of an array of time delays
    model_list (list):  List of instances of the QuantumModel class
    r (float): Renormalization factor applied to avoid small floats in the calculation of probabilities.
    tfin: Final time of the simulation. If -1, final time is set by the time when the last jump in the data array occurs (i.e., tfin = sum(data))
    method: Method used for the simulation. Can be "spectral" or "direct". If "spectral", a spectral decomposition of the Liouvillian is used.

    Returns:
    -----------
    np.array: Unnormalized likelihood function.
    """

    # Compute the likelihood L(D|model) for each model in 'model_list' and store in an array
    likelihood_time_delta = np.asarray([ model.computeLikelihood(data, r, tfin=tfin,method= method) for model in model_list])

    # Normalize the likelihood to obtain a probability distribution P(D|model)
    prob_delta = likelihood_time_delta/np.sum(likelihood_time_delta,axis=0)

    return prob_delta

def compute_log_likelihood_analytical(Delta: float, data: np.array,gamma: float = 1., Omega: float = 1.) -> np.array:
    """
    Returns the log likelihood function log[P(D|Delta)] that will tell us how probable the data D is, given our model with parameters Delta.
    This calculation uses analytical formulas for the likelihood.

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: The log-likelihood function for each observation (we compute it after each jump)
    """
    tau_list = data
    wlist = np.array([compute_waiting_time(tau,Omega,Delta) for tau in tau_list])

    # Jumps are independent, therefore, the likelihood is the product of the probabilities, and the log-likelihood, the sum of the log-probabilities.
    log_likelihood = np.sum(np.log(wlist))+1e-300

    return log_likelihood

def compute_neg_log_likelihood_analytical(delta: float, data: np.array,gamma: float = 1., omega: float = 1.) -> np.array:
    """
    Returns the negative log likelihood function -log[P(D|Delta)].

    This function is a trivial extension of the log likelihood function and returns the negative of the log likelihood

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: The negative log likelihood function for each observation (we compute it after each jump)
    """
        
    return -compute_log_likelihood_analytical(delta, data,gamma, omega)

def compute_likelihood_analytical(Delta: float, data: np.array, gamma: float = 1., Omega: float = 1.) -> np.array:
    """
    Returns the likelihood function P(D|Delta) that will tell us how probable the data D is, given our model with parameters Delta.
    This calculation uses analytical formulas for the likelihood.

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: The likelihood function for each observation (we compute it after each jump)
    """


    log_likelihood =  compute_log_likelihood_analytical(Delta, data, gamma, Omega)

    return np.exp(log_likelihood)

def get_estimates_Bayesian(data: np.array, output_probabilities=False, DeltaMin: float = 0., DeltaMax: float = 4., nDeltaGrid: int = 500):
    """
    Calculate estimates of Delta (1D parameter estimation) in a coherently driven TLS by computing the probability distribution over a grid of Delta values.
    Then, estimates are computed by taking the mean, median and maximum.
    We use analytical computations of the likelihood for this.

    Parameters:
    -----------
    data (np.array): List of time delays representing quantum jumps.
    output_probabilities (bool): Whether to output the probability distribution.
    DeltaMin (float): Minimum value of delta for the analysis.
    DeltaMax (float): Maximum value of delta for the analysis.
    nDeltaGrid (int): Number of points in the grid for delta values.

    Returns:
    -----------
    tuple: A tuple containing the estimated delta values (mean, median, and max).
    If output_probabilities is True:
        The tuple also includes the probability distribution for delta values.

    """
    # Generate a fine grid of delta values
    DeltaBayesListFine = np.linspace(DeltaMin, DeltaMax, nDeltaGrid)

    # Calculate the likelihood grid using analytical computation
    likelihood_grid = np.array([compute_likelihood_analytical(Delta, data) for Delta in DeltaBayesListFine])

    # Compute the probability distribution by normalizing the likelihood
    prob_grid = likelihood_grid / np.sum(likelihood_grid)

    # Calculate mean, median, and maximum estimates for delta
    deltaMean = np.dot(DeltaBayesListFine, prob_grid)
    deltaMedian = DeltaBayesListFine[np.argmin(np.abs(np.cumsum(prob_grid) - 0.5))]
    deltaMax = DeltaBayesListFine[np.argmax(prob_grid)]

    # Check if output_probabilities is requested
    if output_probabilities == False:
        return deltaMean, deltaMedian, deltaMax
    else:
        return deltaMean, deltaMedian, deltaMax, prob_grid

    
def prob_no_click(Delta: float,tau:float,Omega:float =1.):
    """
    Calculates the probability of no jump occurring within a time delay tau after the last jump in a coherently driven two-level system.

    Parameters:
    -----------
    Delta (float): Parameter Delta in units of gamma (i.e., we consider gamma = 1)
    tau (float): Time passed since a jump ocurred, collpasing the system into the ground state
    Omega (float): Parameter Omega in units of gamma (i.e., we consider gamma = 1)

    Returns:
    -----------
    float: Calculated probability
    """
        
    return np.real(
        ((-1/8*1j)*
  (exp(((2 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((-8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (-1 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) + np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
       np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
     4*Delta**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
     16*Omega**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
     np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
       (1 - 4*Delta**2 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))))) + 
   exp(((2 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
        (2*1j)*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((-8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (-1 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) - np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
       np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
     4*Delta**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
     16*Omega**2*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
        np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
     np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
       (1 - 4*Delta**2 - 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))))) + 
   exp(((2 + 1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (1 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) - (4*1j)*Delta**2*
      np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))) - 
     1j*(np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + 
           (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
       16*Omega**2*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
          np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
       np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
         (-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
            8*Delta**2*(1 + 16*Omega**2)))))) + 
   exp(((2 + 2*np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
        1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
           np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))*
       tau)/4)*((8*1j)*np.emath.sqrt(2)*Delta**3 + (2*1j)*np.emath.sqrt(2)*Delta*
      (1 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
         8*Delta**2*(1 + 16*Omega**2))) + (4*1j)*Delta**2*
      np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
          8*Delta**2*(1 + 16*Omega**2))) + 
     1j*(np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + 
           (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) - 
       16*Omega**2*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
          np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))) + 
       np.emath.sqrt((16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))*
         (-1 + 4*Delta**2 + 16*Omega**2 + np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 
            8*Delta**2*(1 + 16*Omega**2))))))))/
 (np.emath.sqrt(2)*Delta*
  exp(((4 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*Delta**2 - 16*Omega**2 + 
         np.emath.sqrt(-64*Omega**2 + (1 + 4*Delta**2 + 16*Omega**2)**2)) + 
      1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*Delta**2 + 16*Omega**2 + 
         np.emath.sqrt(-64*Omega**2 + (1 + 4*Delta**2 + 16*Omega**2)**2)))*tau)/4)*
  np.emath.sqrt(16*Delta**4 + (1 - 16*Omega**2)**2 + 8*Delta**2*(1 + 16*Omega**2))))

def compute_waiting_time_list(params, taulist):
    """
    Calculate the waiting time distribution of a coherently driven two-level system for a list of time delays.

    Parameters:
    ----------
    params (list): List containing two parameters - Omega and Delta.
    taulist (list): List of time delay values.

    Returns:
    ----------
    numpy.ndarray: An array containing the waiting time distribution.
    """
    Omega, Delta = params
    wtau = np.asarray([compute_waiting_time(tau,Omega,Delta) for tau in taulist])
    return wtau


def compute_waiting_time(tau: float,Omega: float,Delta: float):
    """
    Calculates the waiting time distribution of a coherently driven two-level system based on Omega and Delta.

    Parameters:
    ----------
    tau (float): Time delay.
    Omega (float): Parameter Omega.
    Delta (float): Parameter Delta.

    Returns:
    ----------
    float: Calculated w(tau) value.
    """
    w_tau = 8*exp(-tau/2)*Omega**2/sqrt(-64*Omega**2+(1+4*Delta**2+16*Omega**2)**2)*(-cos(tau*sqrt(-1+4*Delta**2+16*Omega**2+sqrt(-64*Omega**2+(1+4*Delta**2+16*Omega**2)**2))/(2*sqrt(2)))
    + cosh(tau*sqrt(1-4*Delta**2-16*Omega**2+sqrt(-64*Omega**2+(1+4*Delta**2+16*Omega**2)**2))/(2*sqrt(2))))

    return w_tau

def compute_population_ss(params: list):
    """
    Calculate the steady-state population of a coherently driven two-level system based on Omega and Delta.

    Parameters:
    ----------
    params (list): List containing two parameters - Omega and Delta.

    Returns:
    ----------
    float: Calculated steady state population 
    """
    Omega, Delta = params
    nsigma = 4*Omega**2/(1**2 + 4*Delta**2 + 8 * Omega**2)
    return nsigma

def compute_likelihood(Delta: float, data: np.array, r: float = 5.0,gamma: float = 1., omega: float = 1.) -> np.array:
    """
    Returns the likelihood function L(D|Delta) proportional to the probability of the data D being measured, given the model parameter Delta.
    This calculation is based on purely numerical calculations using Qutip. The model assumed is a coherently driven two-level system.

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    r (float): Renormalization factor applied to avoid small floats in the calculation of probabilities.
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: Unnormalized likelihood function.
    """

    tau_list = data

    njumps = len(tau_list)
    psi0=basis(2,0)
    a=destroy(2)
    c_ops=[]
    c_ops.append(np.sqrt(gamma)*a)

    # Define system Hamiltonian
    H=Delta*a.dag()*a+omega*a+omega*a.dag()

    L = liouvillian(H,c_ops)
    J = gamma*sprepost(a,a.dag())
    Lhat = L-J
    Uops = [(tau*Lhat).expm() for tau in tau_list]
    rho0 = operator_to_vector(ket2dm(psi0))
    rhoC = rho0

    rhoC_t = []
    for jump_idx  in range(njumps):
        renorm = tau_list[jump_idx]*r
        rhoC=J*Uops[jump_idx]*rhoC
        rhoC = renorm*rhoC
        rhoC_t.append(rhoC)

    likelihoodTime = np.array([vector_to_operator(rhoCsel).tr() for rhoCsel in rhoC_t])   


    return likelihoodTime[-1]


def compute_negative_log_likelihood(Delta: float, data: np.array, r: float = 5.0,gamma: float = 1., omega: float = 1.):
    """
    Returns the negative log-likelihood function -log[L(D|Delta)]

    Parameters:
    -----------
    Delta (float): Model parameter Delta. This is the parameter being estimated.
    data (np.array): The trajectory data in the form of an array of time delays
    r (float): Renormalization factor applied to avoid small floats in the calculation of probabilities.
    gamma (float): Model parameter gamma (asssumed known)
    Omega (float): Model parameter Omega (assumed known)

    Returns:
    -----------
    np.array: Negative log
    """

    return -np.log(compute_likelihood(Delta, data, r, gamma, omega)+1e-300)

def create_directory(filepath:str):
    """
    Looks for a directory, creates it if it is not found.

    Parameters:
    -----------
    filepath (str): Path of the directory

    Returns:
    -----------
    None
    
    """

    if not os.path.exists(filepath):
        print(f"Directory {filepath} not found: creating...")
        os.makedirs(filepath)
    else:
        print("Folder already exists.")
