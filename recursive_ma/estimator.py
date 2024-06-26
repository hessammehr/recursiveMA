from .isotopes import ISOTOPES
from itertools import product
from operator import add, mul
from functools import reduce


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
    
    for child in tree:
        if parent - child < 1.0:
            continue
        subtree1 = find_subtree(tree, child, tol)
        subtree2 = find_subtree(tree, parent - child, tol)
        for p1 in constructions(subtree1, child, tol):
            for p2 in constructions(subtree2, parent - child, tol):
                yield ((parent,) + p1[0] + p2[0], p1[1] + p2[1])
    yield ((parent,), (parent,))

def n_constructions(tree, parent, tol):
    if isinstance(tree, list):
        return reduce(mul, (n_constructions(next(iter(subtree.values())),next(iter(subtree.keys())), tol) for subtree in tree))
    result = 1
    for child in tree:
        if parent - child < 1.0:
            continue
        subtree1 = find_subtree(tree, child, tol)
        subtree2 = find_subtree(tree, parent - child, tol)
        result += n_constructions(subtree1, child, tol) * n_constructions(subtree2, parent - child, tol)
    return result


def joiner(lst, mw, tol):
    if not lst:
        return [mw]
    if mw - lst[-1] < tol:
        return [*lst[:-1], (lst[-1] + mw) / 2]
    return [*lst, mw]

def unique(seq, tol):
    gen = iter(seq)
    last = next(gen)
    yield last
    for item in gen:
        if item - last > tol:
            yield item
            last = item

def overlapping(seq, tol):
    gen = iter(seq)
    last = next(gen)
    for item in seq:
        if item - last < tol:
            yield item
        last = item

def leaf_ma(mw, tol):
    for isotope_mz in ISOTOPES.values():
        if abs(isotope_mz - mw) < tol:
            return 0.0
    return max(0.075 * mw - 1.3, 0.)
    
def construction_ma(construction, tol):
    all_nodes, end_nodes = construction
    joined_all = list(unique(sorted(all_nodes), tol))
    joined_end = list(unique(sorted(end_nodes), tol))
    overlaps = overlapping(sorted(joined_all + joined_end), tol)
    internal = len(joined_all) - len(joined_end)
    return internal + sum(leaf_ma(mw, tol) for mw in joined_end) - sum(leaf_ma(mw, tol) for mw in overlaps)

def estimate_ma(trees, tol):
    trees = [augment(tree, tol) for tree in trees]
    for construction in constructions(trees, None, tol):
        yield construction, construction_ma(construction, tol)

