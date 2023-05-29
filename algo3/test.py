#! venv/bin/python3
import main

gen = main.DataGenerators
size, sat = map(int, input().split())
elst = gen.generate_mixed_circle_egde_list(size)
amatrix = gen.convert_edge_list_into_adjacency_matrix(elst, num_vertices=size)
print(gen.get_vertices_grades(amatrix))
print(amatrix)
graph = gen.add_vertices_by_saturation(amatrix, size, sat)
print(gen.get_vertices_grades(graph))
print(graph)
print(main.Algorithms.is_graph_connected(graph))
print(main.Algorithms.find_euler_cycle(graph))
print(main.Algorithms.find_hamiltonian_cycle(graph))
print(main.Algorithms.find_all_hamiltonian_cycles(graph))

graph_2_test = [
    [0, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0]]

print(gen.get_vertices_grades(graph_2_test))
print(graph_2_test)
print(main.Algorithms.is_graph_connected(graph_2_test))
print(main.Algorithms.find_euler_cycle(graph_2_test))
print(main.Algorithms.find_hamiltonian_cycle(graph_2_test))
print(main.Algorithms.find_all_hamiltonian_cycles(graph_2_test))
