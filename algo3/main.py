#! venv/bin/python3
import numpy as np
import random
from time import process_time_ns
import matplotlib.pyplot as plotter
from datetime import datetime
from sys import setrecursionlimit
from os.path import join as pjoin
from os import mkdir, abort

class DataGenerators:
    @staticmethod
    def generate_mixed_circle_egde_list(num_vertices):
        assert type(num_vertices) is int
        assert num_vertices > 3

        # get starting vertice
        last = random.randint(0, num_vertices - 1)
        first = last

        # create list of available vertices and remove first
        vertices_range = list(range(0, num_vertices))
        vertices_range.remove(last)
        edge_list = []
        while len(vertices_range):
            # chose random vertice and remove its occurance from available vertices list
            random_vertice = random.choice(vertices_range)
            vertices_range.remove(random_vertice)
            # create connection between vertices
            edge_list.append((last, random_vertice))
            last = random_vertice

        # create connection between last and first vertice
        edge_list.append((last, first))
            
        return edge_list
    
    @staticmethod
    def convert_edge_list_into_adjacency_matrix(edge_list, num_vertices):
        assert type(num_vertices) is int
        assert num_vertices > 3

        # convert edge list into adjacency matrix
        adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for edge in edge_list:
            adjacency_matrix[edge[0]][edge[1]] = 1
            adjacency_matrix[edge[1]][edge[0]] = 1
            
        return adjacency_matrix
    
    @staticmethod
    def add_vertices_by_saturation(amatrix, num_vertices, saturation_percentage):
        assert type(num_vertices) is int
        assert num_vertices > 3
        assert type(saturation_percentage) is int
        assert 0 <= saturation_percentage <= 100

        adjacency_matrix = np.copy(amatrix)
        max_edges = int((num_vertices * (num_vertices - 1)) / 2)
        current_saturation = round((num_vertices / max_edges) * 100)
        
        # if current saturation is lower than expected add edges
        if current_saturation < saturation_percentage:
            # calculate how many edges to add
            new_edges = int((((saturation_percentage - current_saturation) / 100)) * ((num_vertices * (num_vertices - 1)) / 2)) + 1

            for _ in range(new_edges):
                vertices_grades = np.sum(adjacency_matrix, axis=1)
                even_vertices = list(range(0, num_vertices))
                odd_grades = []
                for vertice, grade in enumerate(vertices_grades):
                    if grade == (num_vertices - 1):
                        even_vertices.remove(vertice)
                        continue
                    if grade % 2 != 0:
                        odd_grades.append(vertice)
                        even_vertices.remove(vertice)
                
                if len(odd_grades):
                    vertice0 = random.choice(odd_grades)
                else:
                    vertice0 = random.choice(even_vertices)
                    even_vertices.remove(vertice0)

                for vertice, connected in enumerate(adjacency_matrix[vertice0]):
                    if connected and vertice in even_vertices:
                        even_vertices.remove(vertice)

                if len(even_vertices):
                    vertice1 = random.choice(even_vertices)
                else:
                    if vertice0 in odd_grades:
                        odd_grades.remove(vertice0)
                    vertice1 = random.choice(odd_grades)

                adjacency_matrix[vertice0, vertice1] = 1
                adjacency_matrix[vertice1, vertice0] = 1

        return adjacency_matrix
            
    @staticmethod
    def generate_mixed_connected_circle_graph(num_vertices, saturation_percentage):
        assert type(num_vertices) is int
        assert num_vertices > 3
        assert type(saturation_percentage) is int
        assert 0 <= saturation_percentage <= 100

        adjacency_list = DataGenerators.generate_mixed_circle_egde_list(num_vertices)
        adjacency_matrix = DataGenerators.convert_edge_list_into_adjacency_matrix(adjacency_list, num_vertices=num_vertices)
        return DataGenerators.add_vertices_by_saturation(adjacency_matrix, num_vertices, saturation_percentage)
        
    
    @staticmethod
    def get_vertices_grades(imatrix):
        vertices_grades = np.sum(imatrix, axis=1)
        return vertices_grades


class Algorithms:
    @staticmethod

    def find_euler_cycle(m):
        imatrix = np.copy(m)
        num_vertices = imatrix.shape[0]
        cycle = []

        if not Algorithms.is_graph_connected(imatrix):
            print("Graph is not connected.")
            return cycle

        stack = []
        cur = 0
 
        while (len(stack) > 0 or sum(imatrix[cur])!= 0):
            if (sum(imatrix[cur]) == 0):
                cycle.append(cur)
                cur = stack[-1]
                del stack[-1]
            else:
                for i in range(num_vertices):
                    if (imatrix[cur][i] == 1):
                        stack.append(cur)
                        imatrix[cur][i] = 0
                        imatrix[i][cur] = 0
                        cur = i
                        break

        return cycle + [0]

    @staticmethod
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

    @staticmethod
    def find_hamiltonian_cycle(adj_matrix):    
        n = len(adj_matrix)
        visited = [False] * n
        path = []

        start_vertex = 0
        current_vertex = start_vertex
        path.append(current_vertex)
        visited[current_vertex] = True

        while len(path) < n:
            next_vertex = None
            min_distance = float('inf')

            for neighbor in range(n):
                if adj_matrix[current_vertex][neighbor] > 0 and not visited[neighbor]:
                    if adj_matrix[current_vertex][neighbor] < min_distance:
                        next_vertex = neighbor
                        min_distance = adj_matrix[current_vertex][neighbor]

            if next_vertex is None:
                return None

            path.append(next_vertex)
            visited[next_vertex] = True
            current_vertex = next_vertex

        if adj_matrix[path[-1]][path[0]] > 0:
            return path + [start_vertex]
        else:
            return None

    @staticmethod
    def find_all_hamiltonian_cycles(adj_matrix):
        n = len(adj_matrix)
        visited = [False] * n
        cycles = []
    
        def dfs(v, start, path):
            visited[v] = True
            path.append(v)
    
            if len(path) == n:
                # Znaleziono cykl Hamiltona
                if adj_matrix[v][start] == 1:
                    cycles.append(path[:])
    
            for i in range(n):
                if adj_matrix[v][i] == 1 and not visited[i]:
                    dfs(i, start, path)
    
            # Wycofywanie
            visited[v] = False
            path.pop()
    
        for vertex in range(n):
            dfs(vertex, vertex, [])
    
        return cycles

        
class FindingPerformance(DataGenerators, Algorithms):
    def __init__(self):
        super().__init__()
        self.parameters = []
        self.filename = ""
        self.sizes = []
        
    def load_param_from_file(self, filename=None):
        if filename is None:
            filename = input("enter filename\n")
        self.__init__()
        try:
            with open(filename, "r") as file:
                self.filename = filename
                for line in file:
                    self.parameters.append(line.strip())
        except FileNotFoundError:
            self.load_param_from_dialog()

    def load_param_from_dialog(self):
        self.__init__()
        self.parameters.append(int(input("How many sets\n")))
        for _ in range(self.parameters[0]):
            self.parameters.append(input("Enter graph size (how many vertices)\n"))

    def process_parameters(self):
        if len(self.parameters):
            for idx in range(int(self.parameters[0])):
                self.sizes.append(int(self.parameters[idx + 1]))
        else:
            print("Before processing parameters, you should load the data.\t\tAborting...")
            abort()
        
    def Task1(self, saturation):
        assert type(saturation) is int
        assert 0 < saturation <= 100
        perf = {"find_euler_cycle" : [], "find_hamiltonian_cycle": []}
        for i in range(int(self.parameters[0])):
            size = self.sizes[i]
            circle_graph = DataGenerators.generate_mixed_connected_circle_graph(size, saturation)
            for algo in ("find_euler_cycle", "find_hamiltonian_cycle"):
                time = 0
                for _ in range(5):
                    text = f"Performing {algo} measurements - round: {_} - graph size: {size} with {saturation} % of saturation"
                    print(end='\x1b[2K')
                    print(text, end="\r")
                    tstart = process_time_ns()
                    a = getattr(Algorithms, algo)
                    empty = a(circle_graph)
                    tstop = process_time_ns()
                    time += (tstop - tstart)
                avg = round(time / 5) / 10000000
                perf[algo].append(avg)
        print(end='\x1b[2K')
        return perf
                
    def Task2(self, saturation):
        assert type(saturation) is int
        assert 0 < saturation <= 100
        perf = []
        circuits = []
        for i in range(int(self.parameters[0])):
            size = self.sizes[i]
            circle_graph = DataGenerators.generate_mixed_connected_circle_graph(size, saturation)
            text = f"Performing Hamiltonian circuits find measurements - graph size: {size} with {saturation} % of saturation"
            print(end='\x1b[2K')
            print(text, end="\r")
            tstart = process_time_ns()
            empty = Algorithms.find_all_hamiltonian_cycles(circle_graph)
            tstop = process_time_ns()
            perf.append(((tstop - tstart) / 10000000) / 60)
            circuits.append(empty)
        print(end='\x1b[2K')
        return perf, circuits
    

if __name__ == "__main__":
    results_dir = "results"
    try:
        mkdir(results_dir)
    except FileExistsError:
        pass
    
    now_time = datetime.now().strftime('%H%M_%d%m%y')
    dir = pjoin(results_dir, f"{now_time}")
    
    try:
        mkdir(dir)
    except FileExistsError:
        pass
    
    findperf = FindingPerformance()
    findperf.load_param_from_file("algo3")
    findperf.process_parameters()

    task1_30p = findperf.Task1(saturation=30)
    euler_performance = task1_30p["find_euler_cycle"]
    hamiltonian_performance = task1_30p["find_hamiltonian_cycle"]
    
    plotter.plot(list(findperf.sizes), list(euler_performance), color="r", label="find_euler_cycle")
    plotter.plot(list(findperf.sizes), list(hamiltonian_performance), color="g", label="find_hamiltonian_cycle")
    
    plotter.xlabel("graph size [n]", weight='light', style='italic')
    plotter.ylabel("time [s]", weight='light', style='italic')
    plotter.title("Graph saturation 30%", weight='bold')
    plotter.grid('on', linestyle=':', linewidth=0.5)

    plotter.legend()

    file_name = f"task1_30p_plot_{findperf.filename}_{now_time}.png"
    save_path = pjoin(dir, file_name)

    try:
        plotter.savefig(save_path, dpi=500)
    except FileExistsError:
        pass

    plotter.close(None)
    
    task1_70p = findperf.Task1(saturation=70)
    euler_performance = task1_70p["find_euler_cycle"]
    hamiltonian_performance = task1_70p["find_hamiltonian_cycle"]
    
    plotter.plot(list(findperf.sizes), list(euler_performance), color="r", label="find_euler_cycle")
    plotter.plot(list(findperf.sizes), list(hamiltonian_performance), color="g", label="find_hamiltonian_cycle")
    
    plotter.xlabel("graph size [n]", weight='light', style='italic')
    plotter.ylabel("time [s]", weight='light', style='italic')
    plotter.title("Graph saturation 70%", weight='bold')
    plotter.grid('on', linestyle=':', linewidth=0.5)

    plotter.legend()

    file_name = f"task1_70p_plot_{findperf.filename}_{now_time}.png"
    save_path = pjoin(dir, file_name)

    try:
        plotter.savefig(save_path, dpi=500)
    except FileExistsError:
        pass

    plotter.close(None)
    
    findperf.load_param_from_file("algo3_hamiltonian")
    findperf.process_parameters()
    
    task2_50p = findperf.Task2(saturation=50)
    times, cycles = task2_50p[0], task2_50p[1]
    
    plotter.plot(list(findperf.sizes), list(times), color="b", label="finding all hamiltonian circuits")
    
    plotter.xlabel("graph size [n]", weight='light', style='italic')
    plotter.ylabel("time [s]", weight='light', style='italic')
    plotter.title("Graph saturation 50%", weight='bold')
    plotter.grid('on', linestyle=':', linewidth=0.5)

    plotter.legend()

    file_name = f"task2_50p_plot_{findperf.filename}_{now_time}.png"
    save_path = pjoin(dir, file_name) if results_dir is not None else file_name

    try:
        plotter.savefig(save_path, dpi=500)
    except FileExistsError:
        pass

    plotter.close(None)
    
    save_path = pjoin(dir, "all_hamiltonian_circuits.txt") if results_dir is not None else "out.txt"
    with open(save_path, "w") as text_out:
        for i, graph in enumerate(findperf.sizes):
            text_out.write(f"Graph size: {findperf.sizes[i]} with saturation 50 %, Hamiltionian cycles:\n")
            text_out.write(f"{str(cycles[i])}\n")
