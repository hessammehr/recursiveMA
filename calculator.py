import argparse
import pickle
from operator import add
from functools import reduce
from recursive_ma import estimate_ma

usage = """
    python calculator.py <pickle_file> <sample_name1> <sample_name2> ... <tolerance> <iterations>

    <pickle_file> must contain a dictionary with sample names as keys and
    a tree of masses as values.
    """

def main():
    parser = argparse.ArgumentParser(description="Calculate mass accuracy.")
    parser.add_argument("pickle_file", type=str, help="Pickle file with tree of masses.")
    parser.add_argument("sample_names", type=str, nargs="+", help="Sample names.")
    parser.add_argument("tolerance", type=float, help="Tolerance.")
    parser.add_argument("iterations", type=int, help="Number of iterations.")
    args = parser.parse_args()

    with open(args.pickle_file, "rb") as f:
        trees = pickle.load(f)
    all_data = {sample_name: [{ms1_peak: tree} for ms1_peak, tree in sample_data.items()] for sample_name, sample_data in trees.items()}

    data = reduce(add, [all_data[sample_name] for sample_name in args.sample_names])
    optimum = min(zip(estimate_ma(data, args.tolerance), range(args.iterations)), key=lambda c: c[0][1])
    print(optimum[0][1])

if __name__ == "__main__":
    main()