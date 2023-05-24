import numpy as np
import random
from time import perf_counter_ns
import matplotlib.pyplot as plotter
from datetime import datetime
from sys import setrecursionlimit
from os.path import join as pjoin
from os import mkdir, abort

class DataGenerators:
    @staticmethod
    def generate_mixed_circle_incidence_list(num_vertices):
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
            
        return full_incidence_list
    
    @staticmethod
    def convert_incidence_list_into_matrix(incidence_list, num_vertices):
        # convert incidence list into incidence matrix
        incidence_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for edge in incidence_list:
            incidence_matrix[edge[0]][edge[1]] = 1
            
        return incidence_matrix
            
    @staticmethod
    def generate_mixed_connected_circle_graph(num_vertices, saturation_percentage):
        incidence_matrix = DataGenerators.convert_incidence_list_into_matrix(DataGenerators.generate_mixed_circle_incidence_list(num_vertices), num_vertices=num_vertices)

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


class Algorithms:
    @staticmethod

    def find_euler_cycle(m):
        imatrix = np.copy(m)
        num_vertices = len(imatrix)
        cycle = []

        if not Algorithms.is_graph_connected(imatrix):
            print("Graph is not connected.")
            return cycle

        start_vertice = Algorithms.find_start_vertice(imatrix)
        current_vertice = start_vertice

        while True:
            cycle.append(current_vertice)

            if Algorithms.find_next_vertice(imatrix, current_vertice) == -1:
                break

            next_vertice = Algorithms.find_next_vertice(imatrix, current_vertice)

            imatrix[current_vertice][next_vertice] -= 1
            imatrix[next_vertice][current_vertice] -= 1

            current_vertice = next_vertice

        for i in range(num_vertices):
            for j in range(num_vertices):
                if imatrix[i][j] != 0:
                    print("No Euler cycle.")
                    return []

        return cycle

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

    @staticmethod
    def find_next_vertice(imatrix, current_vertice):
        num_vertices = len(imatrix)

        for i in range(num_vertices):
            if imatrix[current_vertice][i] > 0:
                return i

        return -1

    @staticmethod
    def find_hamiltonian_cycle(m):
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
        
        imatrix = np.copy(m)
        num_vertices = len(imatrix)
        cycle = []

        if not Algorithms.is_graph_connected(imatrix):
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
                self.sizes.append(int(self.parameters[idx]))
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
                    tstart = perf_counter_ns()
                    a = getattr(Algorithms, algo)
                    empty = a(circle_graph)
                    tstop = perf_counter_ns()
                    time += (tstop - tstart)
                avg = round(time / 5000)
                perf[algo].append(avg)
        return perf
                
    def Task2(self):
        pass
    

if __name__ == "__main__":
    results_dir = "results"
    try:
        mkdir(results_dir)
    except FileExistsError:
        pass
    findperf = FindingPerformance()
    findperf.load_param_from_file("algo3")
    findperf.process_parameters()
    now_time = datetime.now()

    task1_30p = findperf.Task1(saturation=30)
    euler_performance = task1_30p["find_euler_cycle"]
    hamiltonian_performance = task1_30p["find_hamiltonian_cycle"]
    
    plotter.plot(list(findperf.sizes), list(euler_performance), color="r", label="find_euler_cycle")
    plotter.plot(list(findperf.sizes), list(hamiltonian_performance), color="g", label="find_hamiltonian_cycle")
    
    plotter.xlabel("graph size [n]")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("Graph saturation 30%")

    plotter.legend()

    file_name = f"task1_30p_plot_{findperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png"
    save_path = pjoin(results_dir, file_name) if results_dir is not None else file_name

    try:
        plotter.savefig(save_path)
    except FileExistsError:
        pass

    plotter.close(None)
    
    task1_30p = findperf.Task1(saturation=70)
    euler_performance = task1_30p["find_euler_cycle"]
    hamiltonian_performance = task1_30p["find_hamiltonian_cycle"]
    
    plotter.plot(list(findperf.sizes), list(euler_performance), color="r", label="find_euler_cycle")
    plotter.plot(list(findperf.sizes), list(hamiltonian_performance), color="g", label="find_hamiltonian_cycle")
    
    plotter.xlabel("graph size [n]")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("Graph saturation 70%")

    plotter.legend()

    file_name = f"task1_70p_plot_{findperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png"
    save_path = pjoin(results_dir, file_name) if results_dir is not None else file_name

    try:
        plotter.savefig(save_path)
    except FileExistsError:
        pass

    plotter.close(None)
