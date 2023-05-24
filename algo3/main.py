import numpy as np
import random

def generate_mixed_connected_circle_graph(num_vertices, saturation_percentage):
    # get starting vertice
    last = random.randint(0, num_vertices - 1)
    first = last
    
    # create list of available vertices and remove first
    vertices_range = list(range(0, num_vertices))
    vertices_range.remove(last)
    incidence_list = []
    while len(vertices_range):
        # chose random vertice and remove its occurance from available vertices list
        random_vertice = random.choice(vertices_range)
        vertices_range.remove(random_vertice)
        # create incidence between vertices
        incidence_list.append((last, random_vertice))
        last = random_vertice
        
    # create incidence between last and first vertice
    incidence_list.append((last, first))
    
    # create full incidence list by reversing original list and appending new incidences
    full_incidence_list = []
    full_incidence_list.extend(incidence_list)
    for vertice in incidence_list:
        full_incidence_list.append((vertice[1], vertice[0]))
    
    # convert incidence list into incidence matrix
    incidence_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
    for edge in full_incidence_list:
        incidence_matrix[edge[0]][edge[1]] = 1
        
    max_edges = int((num_vertices * (num_vertices - 1)) / 2)
    current_saturation = round((num_vertices / max_edges) * 100)

    # if current saturation is lower than expected add edges
    if current_saturation < saturation_percentage:
        # calculate how many edges to add
        new_edges = int(((saturation_percentage / 100) - current_saturation) * ((num_vertices * (num_vertices - 1)) / 2))
        
        vertices_grades = np.sum(incidence_matrix, axis=1)
        minimum_vertices_grades = num_vertices * 2
        
        for _ in range(new_edges):
            vertice0 = random.randint(0, num_vertices - 1)
            vertice1 = random.randint(0, num_vertices - 1)

            if incidence_matrix[vertice0, vertice1] == 1 or vertice0 == vertice1 or vertices_grades[vertice0] == minimum_vertices_grades or vertices_grades[vertice1] == minimum_vertices_grades:
                continue

            incidence_matrix[vertice0, vertice1] = 1
            incidence_matrix[vertice1, vertice0] = 1
    
    return incidence_matrix

def find_euler_cycle(m):
    imatrix = np.copy(m)
    num_vertices = len(imatrix)
    cycle = []

    if not is_graph_connected(imatrix):
        print("Graph is not connected.")
        return cycle

    start_vertice = find_start_vertice(imatrix)
    current_vertice = start_vertice

    while True:
        cycle.append(current_vertice)

        if find_next_vertice(imatrix, current_vertice) == -1:
            break

        next_vertice = find_next_vertice(imatrix, current_vertice)

        imatrix[current_vertice][next_vertice] -= 1
        imatrix[next_vertice][current_vertice] -= 1

        current_vertice = next_vertice

    for i in range(num_vertices):
        for j in range(num_vertices):
            if imatrix[i][j] != 0:
                print("No Euler cycle.")
                return []

    return cycle


def is_graph_connected(imatrix):
    num_vertices = len(imatrix)
    visited = [False] * num_vertices
    stack = []

    stack.append(0)
    visited[0] = True
    
    while stack:
        current_vertice = stack.pop()

        for i in range(num_vertices):
            if imatrix[current_vertice][i] > 0 and not visited[i]:
                stack.append(i)
                visited[i] = True

    return all(visited)


def find_start_vertice(imatrix):
    num_vertices = len(imatrix)
    degree = [0] * num_vertices

    for i in range(num_vertices):
        for j in range(num_vertices):
            degree[i] += imatrix[i][j]

    for i in range(num_vertices):
        if degree[i] % 2 != 0:
            return i

    return 0

def find_next_vertice(imatrix, current_vertice):
    num_vertices = len(imatrix)

    for i in range(num_vertices):
        if imatrix[current_vertice][i] > 0:
            return i

    return -1

def find_hamiltonian_cycle(m):
    imatrix = np.copy(m)
    num_vertices = len(imatrix)
    cycle = []

    if not is_graph_connected(imatrix):
        print("Graph is not connected.")
        return cycle

    start_vertice = 0
    cycle.append(start_vertice)
    visited = [False] * num_vertices
    visited[start_vertice] = True

    if find_hamiltonian_cycle_recursive(imatrix, start_vertice, visited, cycle):
        return cycle
    else:
        print("No Hamiltonian cycle.")
        return []


def find_hamiltonian_cycle_recursive(imatrix, current_vertice, visited, cycle):
    num_vertices = len(imatrix)

    if all(visited) and imatrix[current_vertice][cycle[0]] == 1:
        cycle.append(cycle[0])
        return True

    for i in range(num_vertices):
        if imatrix[current_vertice][i] == 1 and not visited[i]:
            cycle.append(i)
            visited[i] = True

            if find_hamiltonian_cycle_recursive(imatrix, i, visited, cycle):
                return True

            cycle.pop()
            visited[i] = False

    return False


num_vertices, saturation = map(int, input().split())
incidence_matrix = generate_mixed_connected_circle_graph(num_vertices, saturation)
if incidence_matrix is not None:
    print(f"euler cycle: {find_euler_cycle(incidence_matrix)}")
    print(f"hamiltonian cycle: {find_hamiltonian_cycle(incidence_matrix)}")