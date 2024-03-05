from qiskit import Aer

backend = Aer.get_backend('qasm_simulator')
print(backend.configuration().to_dict())