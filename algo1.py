from time import perf_counter_ns
from random import randrange
import matplotlib.pyplot as plotter
from datetime import datetime


class SortingAlgorithms:
    def __init__(self):
        self.indata = []
        self.outdata = []

    def insertion_sort(self):
        data = self.indata
        self.outdata = []

        for i in range(1, len(data)):
            key = data[i]
            j = i - 1
            while j >= 0 and key < data[j]:
                data[j + 1] = data[j]
                j -= 1
            data[j + 1] = key
        self.outdata = data

    def selection_sort(self):
        data = self.indata
        self.outdata = []

        for i in range(len(data)):
            min_idx = i
            for j in range(i + 1, len(data)):
                if data[j] < data[min_idx]:
                    min_idx = j
            (data[i], data[min_idx]) = (data[min_idx], data[i])
        self.outdata = data

    def heapsort(self):
        def heapify(d, idx, size):
            l = (2 * idx) + 1
            r = (2 * idx) + 2
            largest = idx
            if l < size and d[l] > d[largest]:
                largest = l
            if r < size and d[r] > d[largest]:
                largest = r
            if largest != idx:
                d[i], d[largest] = d[largest], d[i]
                heapify(d, largest, size)

        data = self.indata
        self.outdata = []

        for i in range(len(data) // 2 - 1, -1, -1):
            heapify(data, i, len(data))

        for i in range(len(data) - 1, -1, -1):
            data[0], data[i] = data[i], data[0]
            heapify(data, 0, i)

        self.outdata = data

    def merge_sort(self):
        def do_merge(d):
            if len(d) > 1:
                p = len(d) // 2
                a1 = d[:p]
                a2 = d[p:]

                do_merge(a1)
                do_merge(a2)

                i = j = k = 0

                while i < len(a1) and j < len(a2):
                    if a1[i] < a2[j]:
                        d[k] = a1[i]
                        i += 1
                    else:
                        d[k] = a2[j]
                        j += 1
                    k += 1

                while i < len(a1):
                    d[k] = a1[i]
                    i += 1
                    k += 1

                while j < len(a2):
                    d[k] = a2[j]
                    j += 1
                    k += 1

        data = self.indata
        self.outdata = []
        do_merge(data)
        self.outdata = data


class DataGenerators:
    @staticmethod
    def random_list(amount, _min, _max, recurring):
        assert amount > 1
        assert type(amount) is int
        assert type(_min) is int
        assert type(_max) is int
        assert recurring.upper() in ["Y", "YES", "N", "NO"]
        nums = []
        for _ in range(amount):
            if recurring.upper() in ("N", "NO") and abs(_max - _min) > amount:
                while True:
                    r = randrange(_min, _max + 1)
                    if r not in nums:
                        nums.append(r)
                        break
            else:
                nums.append(randrange(_min, _max + 1))
        return nums

    @staticmethod
    def ascending_list(amount, _min, diff):
        assert amount > 1
        assert type(amount) is int
        assert type(_min) is int
        assert type(diff) is int
        nums = []
        for idx in range(amount):
            nums.append(
                randrange(_min if idx == 0 else nums[-1], (_min + diff + 1) if idx == 0 else (nums[-1] + diff + 1)))
        return nums

    @staticmethod
    def declining_list(amount, _max, diff):
        assert amount > 1
        assert type(amount) is int
        assert type(_max) is int
        assert type(diff) is int
        nums = []
        for idx in range(amount):
            nums.append(
                randrange((_max - diff) if idx == 0 else (nums[-1] - diff), (_max + 1) if idx == 0 else nums[-1]))
        return nums

    @staticmethod
    def constant_list(amount, value):
        assert amount > 1
        assert type(amount) is int
        assert type(value) is int
        nums = []
        for _ in range(amount):
            nums.append(value)
        return nums

    @staticmethod
    def a_shape_list(amount):
        assert amount > 1
        assert type(amount) is int
        nums = []
        for idx in range(amount // 2):
            nums.append(2 * idx + 1)
        for idx in range(amount // 2, amount):
            nums.append(2 * (amount - idx))
        return nums

    @staticmethod
    def v_shape_list(amount):
        assert amount > 1
        assert type(amount) is int
        nums = []
        for idx in range(amount // 2):
            nums.append((amount // 2 - idx) + 1)
        for idx in range(amount // 2, amount):
            nums.append(2 * (idx - amount // 2))
        return nums


class SortPerformance(SortingAlgorithms, DataGenerators):
    def __init__(self):
        super().__init__()
        self.parameters = []
        self.filename = ""
        self.rnd = []
        self.asc = []
        self.dec = []
        self.con = []
        self.ashape = []
        self.vshape = []
        self.dataset_sizes = []
        self.IS_perf = []
        self.SS_perf = []
        self.HS_perf = []
        self.MS_perf = []

    def load_param_from_file(self, filename=None):
        if filename is None:
            filename = input("enter filename\n")
        try:
            with open(filename, "r") as file:
                self.parameters = []
                self.filename = filename
                for line in file:
                    self.parameters.append(line.strip())
        except FileNotFoundError:
            self.load_param_from_dialog()

    def load_param_from_dialog(self):
        self.parameters = []
        self.parameters.append(input("How many sets of data\n"))
        for _ in range(int(self.parameters[0])):
            self.parameters.append(input("How many to numbers generate?\n"))
            print("random list")
            self.parameters.append(input("Enter minimum and maximum randomness value (min, max)\n"))
            self.parameters.append(input("Numbers can reappear? (y/n)\n"))
            print("ascending list")
            self.parameters.append(input("Enter minimum value\n"))
            self.parameters.append(input("Enter maximum difference between next numbers\n"))
            print("declining list")
            self.parameters.append(input("Enter maximum value\n"))
            self.parameters.append(input("Enter maximum difference between next numbers\n"))
            print("constant list")
            self.parameters.append(input("Enter value\n"))

    def process_parameters(self):
        for idx in range(int(self.parameters[0])):
            idx *= 8
            _min, _max = map(int, self.parameters[2 + idx].split(", "))
            self.rnd.append(self.random_list(int(self.parameters[1 + idx]), _min, _max, self.parameters[3 + idx]))
            self.asc.append(self.ascending_list(int(self.parameters[1 + idx]), int(self.parameters[4 + idx]),
                                                int(self.parameters[5 + idx])))
            self.dec.append(self.declining_list(int(self.parameters[1 + idx]), int(self.parameters[6 + idx]),
                                                int(self.parameters[7 + idx])))
            self.con.append(self.constant_list(int(self.parameters[1 + idx]), int(self.parameters[8 + idx])))
            self.ashape.append(self.a_shape_list(int(self.parameters[1 + idx])))
            self.vshape.append(self.v_shape_list(int(self.parameters[1 + idx])))
            self.dataset_sizes.append(int(self.parameters[1 + idx]))

    def measure_insertion_sort_performance(self):
        self.IS_perf = []
        print(f"executing insertion sort measurements")
        for dataset in range(int(self.parameters[0])):
            print(f"dataset: {dataset}")
            avg_time_ms = []
            ds = ("random", "ascending", "declining", "constant", "vshape")
            for i, d in enumerate(
                    (self.rnd[dataset], self.asc[dataset], self.dec[dataset], self.con[dataset], self.vshape[dataset])):
                time = 0
                print(f"dataset type: {ds[i]}")
                for _ in range(5):
                    print(f"round: {_}")
                    self.indata = d
                    tstart = perf_counter_ns()
                    self.insertion_sort()
                    tstop = perf_counter_ns()
                    time += (tstop - tstart)
                avg = round(time / 5000)
                print(f"insertion sort elapsed time: {avg} ms for {ds[i]} dataset type")
                avg_time_ms.append(avg)
            self.IS_perf.append(avg_time_ms)

    def measure_selection_sort_performance(self):
        self.SS_perf = []
        print(f"executing selection sort measurements")
        for dataset in range(int(self.parameters[0])):
            print(f"dataset: {dataset}")
            avg_time_ms = []
            ds = ("random", "ascending", "declining", "constant", "vshape")
            for i, d in enumerate(
                    (self.rnd[dataset], self.asc[dataset], self.dec[dataset], self.con[dataset], self.vshape[dataset])):
                time = 0
                print(f"dataset type: {ds[i]}")
                for _ in range(5):
                    print(f"round: {_}")
                    self.indata = d
                    tstart = perf_counter_ns()
                    self.selection_sort()
                    tstop = perf_counter_ns()
                    time += (tstop - tstart)
                avg = round(time / 5000)
                print(f"selection sort elapsed time: {avg} ms for {ds[i]} dataset type")
                avg_time_ms.append(avg)
            self.SS_perf.append(avg_time_ms)

    def measure_heapsort_performance(self):
        self.HS_perf = []
        print(f"executing heapsort measurements")
        for dataset in range(int(self.parameters[0])):
            print(f"dataset: {dataset}")
            avg_time_ms = []
            ds = ("random", "ascending", "declining", "constant", "vshape")
            for i, d in enumerate(
                    (self.rnd[dataset], self.asc[dataset], self.dec[dataset], self.con[dataset], self.vshape[dataset])):
                time = 0
                print(f"dataset type: {ds[i]}")
                for _ in range(5):
                    print(f"round: {_}")
                    self.indata = d
                    tstart = perf_counter_ns()
                    self.heapsort()
                    tstop = perf_counter_ns()
                    time += (tstop - tstart)
                avg = round(time / 5000)
                print(f"heapsort elapsed time: {avg} ms for {ds[i]} dataset type")
                avg_time_ms.append(avg)
            self.HS_perf.append(avg_time_ms)

    def measure_merge_sort_performance(self):
        self.MS_perf = []
        print(f"executing merge sort measurements")
        for dataset in range(int(self.parameters[0])):
            print(f"dataset: {dataset}")
            avg_time_ms = []
            ds = ("random", "ascending", "declining", "constant", "vshape")
            for i, d in enumerate(
                    (self.rnd[dataset], self.asc[dataset], self.dec[dataset], self.con[dataset], self.vshape[dataset])):
                time = 0
                print(f"dataset type: {ds[i]}")
                for _ in range(5):
                    print(f"round: {_}")
                    self.indata = d
                    tstart = perf_counter_ns()
                    self.merge_sort()
                    tstop = perf_counter_ns()
                    time += (tstop - tstart)
                avg = round(time / 5000)
                print(f"merge sort elapsed time: {avg} ms for {ds[i]} dataset type")
                avg_time_ms.append(avg)
            self.MS_perf.append(avg_time_ms)


if __name__ == "__main__":
    sortperf = SortPerformance()
    sortperf.load_param_from_file()
    sortperf.process_parameters()
    now_time = datetime.now()

    sortperf.measure_insertion_sort_performance()
    sortperf.measure_selection_sort_performance()
    sortperf.measure_heapsort_performance()
    sortperf.measure_merge_sort_performance()

    IS_rnd = (i[0] for i in sortperf.IS_perf)
    IS_asc = (i[1] for i in sortperf.IS_perf)
    IS_dec = (i[2] for i in sortperf.IS_perf)
    IS_con = (i[3] for i in sortperf.IS_perf)
    IS_vs = (i[4] for i in sortperf.IS_perf)

    SS_rnd = (i[0] for i in sortperf.SS_perf)
    SS_asc = (i[1] for i in sortperf.SS_perf)
    SS_dec = (i[2] for i in sortperf.SS_perf)
    SS_con = (i[3] for i in sortperf.SS_perf)
    SS_vs = (i[4] for i in sortperf.SS_perf)

    HS_rnd = (i[0] for i in sortperf.HS_perf)
    HS_asc = (i[1] for i in sortperf.HS_perf)
    HS_dec = (i[2] for i in sortperf.HS_perf)
    HS_con = (i[3] for i in sortperf.HS_perf)
    HS_vs = (i[4] for i in sortperf.HS_perf)

    MS_rnd = (i[0] for i in sortperf.MS_perf)
    MS_asc = (i[1] for i in sortperf.MS_perf)
    MS_dec = (i[2] for i in sortperf.MS_perf)
    MS_con = (i[3] for i in sortperf.MS_perf)
    MS_vs = (i[4] for i in sortperf.MS_perf)

    plotter.plot(list(sortperf.dataset_sizes), list(IS_rnd), color="r", label="insertion sort")
    plotter.plot(list(sortperf.dataset_sizes), list(SS_rnd), color="g", label="selection sort")
    plotter.plot(list(sortperf.dataset_sizes), list(HS_rnd), color="b", label="heapsort")
    plotter.plot(list(sortperf.dataset_sizes), list(MS_rnd), color="k", label="merge sort")

    plotter.xlabel("dataset size")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("Random data")

    plotter.legend()

    try:
        plotter.savefig(f"RND_plot_{sortperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png")
    except FileExistsError:
        pass

    plotter.close(None)

    plotter.plot(list(sortperf.dataset_sizes), list(IS_asc), color="r", label="insertion sort")
    plotter.plot(list(sortperf.dataset_sizes), list(SS_asc), color="g", label="selection sort")
    plotter.plot(list(sortperf.dataset_sizes), list(HS_asc), color="b", label="heapsort")
    plotter.plot(list(sortperf.dataset_sizes), list(MS_asc), color="k", label="merge sort")

    plotter.xlabel("dataset size")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("Ascending data")

    plotter.legend()

    try:
        plotter.savefig(f"ASC_plot_{sortperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png")
    except FileExistsError:
        pass

    plotter.close(None)

    plotter.plot(list(sortperf.dataset_sizes), list(IS_dec), color="r", label="insertion sort")
    plotter.plot(list(sortperf.dataset_sizes), list(SS_dec), color="g", label="selection sort")
    plotter.plot(list(sortperf.dataset_sizes), list(HS_dec), color="b", label="heapsort")
    plotter.plot(list(sortperf.dataset_sizes), list(MS_dec), color="k", label="merge sort")

    plotter.xlabel("dataset size")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("Declining data")

    plotter.legend()

    try:
        plotter.savefig(f"DEC_plot_{sortperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png")
    except FileExistsError:
        pass

    plotter.close(None)

    plotter.plot(list(sortperf.dataset_sizes), list(IS_con), color="r", label="insertion sort")
    plotter.plot(list(sortperf.dataset_sizes), list(SS_con), color="g", label="selection sort")
    plotter.plot(list(sortperf.dataset_sizes), list(HS_con), color="b", label="heapsort")
    plotter.plot(list(sortperf.dataset_sizes), list(MS_con), color="k", label="merge sort")

    plotter.xlabel("dataset size")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("Constant data")

    plotter.legend()

    try:
        plotter.savefig(f"CON_plot_{sortperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png")
    except FileExistsError:
        pass

    plotter.close(None)

    plotter.plot(list(sortperf.dataset_sizes), list(IS_vs), color="r", label="insertion sort")
    plotter.plot(list(sortperf.dataset_sizes), list(SS_vs), color="g", label="selection sort")
    plotter.plot(list(sortperf.dataset_sizes), list(HS_vs), color="b", label="heapsort")
    plotter.plot(list(sortperf.dataset_sizes), list(MS_vs), color="k", label="merge sort")

    plotter.xlabel("dataset size")
    plotter.ylabel("time [ms]")
    plotter.yscale("log")
    plotter.title("V-shaped data")

    plotter.legend()

    try:
        plotter.savefig(f"VS_plot_{sortperf.filename}_{now_time.strftime('%H%M_%d%m%y')}.png")
    except FileExistsError:
        pass

    plotter.close(None)
