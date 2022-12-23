import cirq
from cirq import Simulator
from cirq.contrib.svg import SVGCircuit


def xx_gate(qubit1, qubit2):
    return cirq.Circuit(cirq.CNOT(qubit1, qubit2), cirq.rz(0.1).on(qubit2), cirq.CNOT(qubit1, qubit2))
def xy_gate(qubit1, qubit2):
    return cirq.Circuit(cirq.CNOT(qubit1, qubit2), cirq.rz(0.1).on(qubit2), cirq.CNOT(qubit1, qubit2))

    
def ansatz_to_circuit(ansatz, qubits, theta):
    circuit = cirq.Circuit()
    #make xx gate
    for i in range(len(qubits)-1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        circuit.append(cirq.rz(theta[i]).on(qubits[i+1]))
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1])) 
    for i in range(len(ansatz)):
        circuit.append(ansatz[i].on(qubits[i]))
        circuit.append(cirq.rz(theta[i]).on(qubits[i]))
    print(circuit)
    return circuit

ansatz = [cirq.X, cirq.Y, cirq.Z]
qubits = cirq.LineQubit.range(3)
theta = [0.1, 0.2, 0.3]
ansatz_to_circuit(ansatz, qubits, theta)