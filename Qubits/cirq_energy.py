import cirq
from Func import *
import numpy as np


def cirq_Energy(theta, N, cirq_ansatz, cirq_Hamiltonian, K, HVA = False):
    if HVA:
        theta = distribute_over_gates(HVA, N, theta)

    #create circuit
    theta = np.array(theta)+np.pi*np.array(K)/4 #Add K to the theta
    #cirq_ansatz = [np.exp(1j*theta[i]*cirq_ansatz[i]) for i in range(len(theta))]
    cirq_ansatz = [cirq.PauliSumExponential(cirq_ansatz[i],theta[i]) for i in range(len(theta))]
    circuit = cirq.Circuit(cirq_ansatz)

    #evaluate the circuit
    simulator = cirq.Simulator(dtype = np.complex128)
    #calclate hamiltonian
    return simulator.simulate_expectation_values(circuit, cirq_Hamiltonian)[0].real


