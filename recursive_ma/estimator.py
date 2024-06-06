# %%
ISOTOPES = {
    # "Antimony": 120.903824,
    # "Argon": 39.962383,
    # "Arsenic": 74.921596,
    # "Barium": 137.905236,
    # "Bismuth": 208.980388,
    "Bromine": 78.918336,
    # "Cadmium": 113.903361,
    # "Calcium": 39.962591,
    # "Cerium": 139.905442,
    # "Cesium": 132.905433,
    "Chlorine": 34.968853,
    # "Chromium": 51.94051,
    # "Cobalt": 58.933198,
    # "Copper": 62.929599,
    # "Dysprosium": 163.929183,
    # "Erbium": 165.930305,
    # "Europium": 152.921243,
    # "Gadolinium": 157.924111,
    # "Gallium": 68.925581,
    # "Germanium": 73.921179,
    # "Gold": 196.96656,
    # "Hafnium": 179.946561,
    # "Holmium": 164.930332,
    # "Indium": 114.903875,
    "Iodine": 126.904477,
    # "Iridium": 192.962942,
    # "Iron": 55.934939,
    # "Krypton": 83.911506,
    # "Lanthanum": 138.906355,
    # "Lead": 207.976641,
    # "Lutetium": 174.940785,
    # "Manganese": 54.938046,
    # "Mercury": 201.970632,
    # "Molybdenum": 97.905405,
    # "Neodymium": 141.907731,
    # "Nickel": 57.935347,
    # "Niobium": 92.906378,
    # "Osmium": 191.961487,
    # "Palladium": 105.903475,
    # "Platinum": 194.964785,
    "Potassium": 38.963708,
    # "Praseodymium": 140.907657,
    # "Rhenium": 186.955765,
    # "Rhodium": 102.905503,
    # "Rubidium": 84.9118,
    # "Ruthenium": 101.904348,
    # "Samarium": 151.919741,
    # "Scandium": 44.955914,
    # "Selenium": 79.916521,
    "Silver": 106.905095,
    # "Strontium": 87.905625,
    "Sulfur": 33.967868,
    # "Tantalum": 180.948014,
    # "Tellurium": 129.906229,
    # "Terbium": 158.92535,
    # "Thallium": 204.97441,
    # "Thorium": 232.038054,
    # "Thulium": 168.934225,
    "Tin": 119.902199,
    # "Titanium": 47.947947,
    # "Tungsten": 183.950953,
    # "Uranium": 238.050786,
    # "Vanadium": 50.943963,
    # "Xenon": 131.904148,
    # "Ytterbium": 173.938873,
    # "Yttrium": 88.905856,
    "Zinc": 63.929145,
    # "Zirconium": 89.904708,
}


# %%
from itertools import product
from operator import add
from functools import reduce
from tqdm.auto import tqdm


def unify_trees(trees: list[dict]):
    """
    Recursively merge `trees` into one tree.
    """
    if not trees:
        return {}
    elif len(trees) == 1:
        return trees[0]
    else:
        child1, child2, *rest = trees
        child1_keys = set(child1 or {})
        child2_keys = set(child2 or {})
        common_keys = child1_keys.intersection(child2_keys)
        return {
            **{k: child1[k] for k in child1_keys - common_keys},
            **{k: child2[k] for k in child2_keys - common_keys},
            **{k: unify_trees([child1[k], child2[k]]) for k in common_keys},
        }


def augment(tree, tol):
    if not tree:
        return tree
    augmented_subtrees = {
        child: augment(subtree, tol) for child, subtree in tree.items()
    }
    for child, subtree in augmented_subtrees.items():
        for grandchild in tree:
            for complement in tree:
                if complement > grandchild:
                    continue
                if abs(child - grandchild - complement) < tol:
                    augmented_subtrees[child] = {
                        **subtree,
                        grandchild: unify_trees(
                            [tree[grandchild], subtree.get(grandchild, {})]
                        ),
                        complement: unify_trees(
                            [tree[complement], subtree.get(complement, {})]
                        ),
                    }
    return augmented_subtrees


def find_subtree(tree, child, tol):
    return unify_trees([subtree for c, subtree in tree.items() if abs(c - child) < tol])


def constructions(tree, parent, tol):
    if isinstance(tree, list):
        consts = [
            constructions(next(iter(subtree.values())), next(iter(subtree.keys())), tol)
            for subtree in tree
        ]
        for c in product(*consts):
            yield (
                reduce(add, (tup[0] for tup in c), ()),
                reduce(add, (tup[1] for tup in c), ()),
            )
        return
    tree = augment(tree, tol)
    yield ((parent,), (parent,))
    for child, subtree in tree.items():
        subtree1 = find_subtree(tree, child, tol)
        subtree2 = find_subtree(tree, parent - child, tol)
        for p1 in constructions(subtree1, child, tol):
            for p2 in constructions(subtree2, parent - child, tol):
                yield ((parent,) + p1[0] + p2[0], p1[1] + p2[1])


def joiner(lst, mw, tol):
    if not lst:
        return [mw]
    if mw - lst[-1] < tol:
        return [*lst[:-2], (lst[-1] + mw) / 2]
    return [*lst, mw]

def leaf_ma(mw, tol):
    for isotope_mz in ISOTOPES.values():
        if abs(isotope_mz - mw) < tol:
            return 0.0
    return max(0.075 * mw - 1.3, 0.)
    
def ma(construction, tol):
    all_nodes, end_nodes = construction
    joined_all = reduce(lambda lst, mw: joiner(lst, mw, tol), sorted(all_nodes), [])
    joined_end = reduce(lambda lst, mw: joiner(lst, mw, tol), sorted(end_nodes), [])
    internal = len(joined_all) - len(joined_end)
    return internal + sum(leaf_ma(mw, tol) for mw in joined_end)

# %%
test_data = {450.0: {120.0: {}, 300.0: {}, 80.0: {}, 220.0: {}, 51.0: {}, 29.01: {}}}
test_data2 = {
    647.8: {
        102.1: {},
    }
}
import random
test_data_big = [
    {random.random() * 200.0 + 250.0: {random.random() * 200.0 + 50.0: {} for _ in range(15)}}
    for _ in range(100)
    # {1150.0: {random.random() * 400.0 + 50.0: {} for _ in range(100)}},
]
joint_data = [test_data, test_data2]
# test_data = {450.0: {150: {120:{}, 60:{}}, 300: {150: {}, 120:{}}}}
# %%
test_agumented = augment(test_data, 0.1)
test_agumented2 = augment(test_data2, 0.1)
joint_agumented = [test_agumented, test_agumented2]
# %%
test_data
# %%
list(constructions(test_data[450.0], 450.0, 0.1))
# %%
list(constructions(test_agumented[450.0], 450.0, 0.1))
# %%
list(constructions(joint_data, None, 0.1))

# %%
list(constructions(joint_agumented, None, 0.1))

# %%
{c: ma(c, 0.1) for c in constructions(joint_agumented, None, 0.1)}
# %%
{c: ma(c, 0.1) for c in constructions(joint_data, None, 0.1)}

# %%
{c: ma(c, 0.1) for c in constructions(test_data_big, None, 0.1)}

# %%
%%time
optimum = min(constructions(test_data_big, None, 0.1), key=lambda c: ma(c, 0.1))
ma(optimum, 0.1)

# %%
individuals = [min(constructions([data], None, 0.1), key=lambda c: ma(c, 0.1)) for data in test_data_big]
x = {c: ma(c, 0.1) for c in individuals}
x, sum(x.values())

# %%
%%time
optimum = min(zip(constructions(test_data_big, None, 0.1), range(1500000)), key=lambda c: ma(c[0], 0.1))[0]
optimum, ma(optimum, 0.1)
# %%
