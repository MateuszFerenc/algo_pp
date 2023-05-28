#! venv/bin/python3
import main

gen = main.DataGenerators
size, sat = map(int, input().split())
ilst = gen.generate_mixed_circle_incidence_list(size)
imatrix = gen.convert_incidence_list_into_matrix(ilst, num_vertices=size)
print(gen.get_vertices_grades(imatrix))
print(imatrix)
graph = gen.add_vertices_by_saturation(imatrix, size, sat)
print(gen.get_vertices_grades(graph))
print(graph)
print(main.Algorithms.is_graph_connected(graph))
print(main.Algorithms.find_euler_cycle(graph))
print(main.Algorithms.find_hamiltonian_cycles(graph))
