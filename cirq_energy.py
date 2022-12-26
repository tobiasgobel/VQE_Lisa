import cirq
import numpy as np

def cirq_Energy(theta, cirq_ansatz, cirq_Hamiltonian, K):

    #create circuit
    theta = np.array(theta)+np.pi*K/4 #Add the K to the theta
    cirq_ansatz = [cirq.PauliSumExponential(cirq_ansatz[i],theta[i]) for i in range(len(theta))]
    circuit = cirq.Circuit(cirq_ansatz)

    #evaluate the circuit
    simulator = cirq.Simulator()
    return simulator.simulate_expectation_values(circuit, cirq_Hamiltonian)[0].real


