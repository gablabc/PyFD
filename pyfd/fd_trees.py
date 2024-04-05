from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from abc import ABC, abstractmethod
from heapq import heappush, heappop

from .decompositions import get_h_add


SYMBOLS = { True :
            {"leq" : "$\,\leq\,$",
            "and_str" : "$)\,\,\land$\,\,(",
            "up" : "$\,>\,$",
            "low" : "$\,<\,$",
            "in_set" : "$\in$"},
            False : 
            {"leq" : "≤",
            "and_str" : " & ",
            "up" : ">",
            "low": "<",
            "in_set" : "∈"}
        }


class Node(object):
    """ Node in a Decision Tree """
    def __init__(self, instances_idx, depth, loa):
        self.N_samples = len(instances_idx)
        self.depth = depth
        self.loa = loa
        # Placeholders
        self.feature = None
        self.threshold = None
        self.child_left = None
        self.child_right = None
        self.splits = []
        self.objectives = []


    def update(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold




class FDTree(BaseEstimator, ABC):
    """ Train a binary tree to minimize : LoA + alpha |L| """
    def __init__(self, features, 
                 max_depth=3,
                 min_samples_leaf=20,
                 branching_per_node=1,
                 alpha=0.05,
                 save_losses=False):
        """
        Parameters
        ----------
        features : Feature object

        max_depth : int, default=3
            Maximum depth of FDTrees

        min_samples_leaf : int, default=20
            The minimum number of samples allowed per leaf

        branching_per_node : int, default=1
            At each node, we consider `branching_per_node` different splits candidates. A value of
            `1` corresponds to greedy CART-like optimization, while larger values allow to try various
            splits and return a more optimal solution. Note that the training scales as `O(branching_per_node^max_depth)`

        alpha : float, default=0.05
            Regularization of the training objective `LoA + alpha |L|`. That is, splitting nodes increases the loss by `alpha`
            and so the reduction in `LoA` must be large enough to compensate.
        """
        self.feature_objs = features.feature_objs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.branching_per_node = branching_per_node
        self.alpha = alpha
        self.save_losses = save_losses


    def print(self, verbose=False, return_string=False):
        tree_strings = []
        self.group_idx = 0
        self.recurse_print_tree_str(self.root, verbose=verbose, tree_strings=tree_strings)
        tree_strings.append(f"Final LoA {self.final_loa:.4f}")
        if return_string:
            return "\n".join(tree_strings)
        else:
            print("\n".join(tree_strings))

    
    def recurse_print_tree_str(self, node, verbose=False, tree_strings=[]):
        if verbose:
            tree_strings.append("|   " * node.depth + f"LoA {node.loa:.4f}")
            tree_strings.append("|   " * node.depth + f"Samples {node.N_samples:d}")
        # Leaf
        if node.child_left is None:
            tree_strings.append("|   " * node.depth + f"Group {self.group_idx}")
            self.group_idx += 1
        # Internal node
        else:
            curr_feature_name = self.feature_objs[node.feature].name
            tree_strings.append("|   " * node.depth + f"If {curr_feature_name} ≤ {node.threshold:.4f}:")
            self.recurse_print_tree_str(node=node.child_left, verbose=verbose, tree_strings=tree_strings)
            tree_strings.append("|   " * node.depth + "else:")
            self.recurse_print_tree_str(node=node.child_right, verbose=verbose, tree_strings=tree_strings)


    def get_split_candidates(self, x_i, i):
        """ Return a list of split candiates """
        if self.feature_objs[i].type == "num":
            x_i_unique = np.unique(x_i)
            if len(x_i_unique) < 40:
                splits = np.sort(x_i_unique)[:-1:]
            else:
                splits = np.quantile(x_i, np.arange(1, 40) / 40)
                # It is possible that quantiles equal the last element when there are
                # duplications. Hence we remove those splits to avoid leaves with no data
                splits = splits[~np.isclose(splits, np.max(x_i))]
        elif self.feature_objs[i].type == "sparse_num":
            is_nonzero = np.where(x_i > 0)[0]
            if len(is_nonzero) == 0:
                splits = []
            else:
                x_i_nonzero = x_i[is_nonzero]
                x_i_non_zero_unique = np.unique(x_i_nonzero)
                if len(x_i_non_zero_unique) < 50:
                    splits = np.sort(x_i_non_zero_unique)[::-1]
                else:
                    splits = np.append(0, np.quantile(x_i_nonzero, np.arange(1, 50) / 50))
                # It is possible that quantiles equal the last element when there are
                # duplications. Hence we remove those splits to avoid leaves with no data
                splits = splits[~np.isclose(splits, np.max(x_i))]
        # Integers we take the values directly
        elif self.feature_objs[i].type in ["ordinal", "num_int"]:
            splits = np.sort(np.unique(x_i))[:-1]
        elif self.feature_objs[i].type == "bool":
            x_i = np.unique(x_i)
            if len(x_i) == 1:
                splits = []
            else:
                splits = np.array([0])
        else:
            raise Exception("Nominal features are not yet supported")

        return splits
    

    def get_feature_splits_heapq(self, curr_node, instances_idx):
        """ Compute a heapqueue for splits along each feature """
        heapq = []
        for feature in range(self.D):
            splits, loa_left, loa_right = self.get_objective_for_splits(instances_idx, feature)
            # No split was conducted
            if len(splits) == 0:
                pass
            else:
                
                # Otherwise search for the best split
                objective = loa_right + loa_left
                if self.save_losses:
                    curr_node.splits.append(splits)
                    curr_node.objectives.append(objective/len(instances_idx))
                
                best_split_idx = np.argmin(objective)
                # The heap contains (obj, feature, split_value, obj_left, obj_right)
                heappush(heapq, (objective[best_split_idx], feature, splits[best_split_idx],
                                loa_left[best_split_idx] / self.N, loa_right[best_split_idx] / self.N))
    
        return heapq
    

    @abstractmethod
    def get_objective_for_splits(self, instances_idx, feature):
        """ Get the objective value at each split """
        pass
    

    @abstractmethod
    def fit(self, X, **kwargs):
        """ Fit the tree via the provided tensors """
        self.loa_factor = None # To have an loa [0, 1]
        self.n_groups = 0
        pass

    
    def _tree_builder(self, instances_idx, depth, loa):
        
        # Create a node
        curr_node = Node(instances_idx, depth, loa*self.loa_factor)

        # Subobjective at that node
        node_loa = loa * self.loa_factor

        # Stop the tree growth if the maximum depth is attained, 
        # or not further split can be justified given the regularization alpha, 
        # or any further split will yield leaves with too few samples
        if depth >= self.max_depth or node_loa < self.alpha or len(instances_idx) < 2 * self.min_samples_leaf:
            return curr_node, node_loa + self.alpha, 1
        
        # Otherwise get a heapq of split candidates
        heapq = self.get_feature_splits_heapq(curr_node, instances_idx)

        # Stop the tree growth if no further splits are
        if len(heapq) == 0:
            return curr_node, node_loa + self.alpha, 1

        # If the next split is guaranteed to be a leaf, we do not need to branch
        if depth + 1 == self.max_depth:
            n_branching = 1
        else:
            n_branching = min(len(heapq), self.branching_per_node)
        subobjective_per_branch = np.zeros(n_branching)
        nodes_per_branch = [deepcopy(curr_node) for _ in range(n_branching)]
        n_leaves_per_branch = np.zeros(n_branching, dtype=np.int32)
        for branch in range(n_branching):

            _, feature_split, split_value, loa_left, loa_right = heappop(heapq)

            # Select instances of the chosen feature
            x_i = self.X[instances_idx, feature_split]

            # Update the node
            nodes_per_branch[branch].update(feature_split, split_value)

            # Go left
            nodes_per_branch[branch].child_left, subobjective_left, n_leaves_left = \
                                self._tree_builder(instances_idx[x_i <= split_value],
                                                    depth=depth+1, loa=loa_left)
            # Go right
            nodes_per_branch[branch].child_right, subobjective_right, n_leaves_right = \
                                self._tree_builder(instances_idx[x_i > split_value],
                                                    depth=depth+1, loa=loa_right)
            n_leaves_per_branch[branch] = n_leaves_left + n_leaves_right
            subobjective_per_branch[branch] = subobjective_left + subobjective_right
        
        # Identify the best branch from the current node
        best_branch = np.argmin(subobjective_per_branch)

        # The best solution resulting from branching has introduced many leaves. Hence, if the reduction in LoA is
        # not sufficient, it is best not to split and define the current node as a leaf
        if node_loa + self.alpha <= subobjective_per_branch[best_branch]:
            return curr_node, node_loa + self.alpha, 1
        else:
            # Branching lead to a better solution and so we go up in the recursion
            return nodes_per_branch[best_branch], subobjective_per_branch[best_branch], n_leaves_per_branch[best_branch]


    def predict(self, X_new):
        """ Return the group index of each instance """
        groups = np.zeros(X_new.shape[0], dtype=np.int32)
        self.group_idx = 0
        if self.n_groups == 1:
            return groups
        else:
            self._tree_traversal_predict(self.root, np.arange(X_new.shape[0]), X_new, groups)
            return groups


    def _tree_traversal_predict(self, node, instances_idx, X_new, groups):
        
        if node.child_left is None:
            # Label the instances at the leaf
            groups[instances_idx] = self.group_idx
            self.group_idx += 1
        else:
            x_i = X_new[instances_idx, node.feature]
            # Go left
            self._tree_traversal_predict(node.child_left, 
                                         instances_idx[x_i <= node.threshold],
                                         X_new, groups)
            
            # Go right
            self._tree_traversal_predict(node.child_right, 
                                         instances_idx[x_i > node.threshold],
                                         X_new, groups)


    def rules(self, use_latex=False):
        """ Return the rule for each leaf """
        self.group_idx = 0
        if self.n_groups == 1:
            return "all"
        else:
            rules = {}
            curr_rule = []
            self._tree_traversal_rules(self.root, rules, curr_rule, use_latex)
            return rules


    def _tree_traversal_rules(self, node, rules, curr_rule, use_latex):
        
        if node.child_left is None:
            if len(curr_rule) > 1:
                # Simplify long rule lists if possible
                curr_rule_copy = self.postprocess_rules(curr_rule, use_latex)
                if len(curr_rule_copy) > 1:
                    rules[self.group_idx] = "(" + SYMBOLS[use_latex]["and_str"].join(curr_rule_copy) + ")"
                else:
                    rules[self.group_idx] = curr_rule_copy[0]
            else:
                rules[self.group_idx] = curr_rule[0]
            self.group_idx += 1
        else:

            feature_obj = self.feature_objs[node.feature]
            feature_name = feature_obj.name
            feature_type = feature_obj.type

            # Boolean
            if feature_type == "bool":
                assert np.isclose(node.threshold, 0)
                curr_rule.append(f"not {feature_name}")
            # Ordinal
            elif feature_type == "ordinal":
                categories = np.array(feature_obj.cats)
                cats_left = categories[:int(node.threshold)+1]
                if len(cats_left) == 1:
                    curr_rule.append(f"{feature_name}={cats_left[0]}")
                else:
                    curr_rule.append(f"{feature_name} " + SYMBOLS[use_latex]['in_set'] \
                                     + " [" + ",".join(cats_left)+"]")
            # Numerical
            else:
                curr_rule.append(feature_name + SYMBOLS[use_latex]['leq'] +\
                                f"{node.threshold:.2f}")
            

            # Go left
            self._tree_traversal_rules(node.child_left, rules, curr_rule, use_latex)
            curr_rule.pop()


            # Boolean
            if feature_type == "bool":
                curr_rule.append(f"{feature_name}")
            # Ordinal
            elif feature_type == "ordinal":
                cats_right = categories[int(node.threshold)+1:]
                if len(cats_right) == 1:
                    curr_rule.append(f"{feature_name}={cats_right[0]}")
                else:
                    curr_rule.append(f"{feature_name} " + SYMBOLS[use_latex]['in_set'] \
                                     + " [" + ",".join(cats_right)+"]")
            # Numerical
            else:
                curr_rule.append(feature_name + SYMBOLS[use_latex]['up'] +\
                                f"{node.threshold:.2f}")
            
            # Go right
            self._tree_traversal_rules(node.child_right, rules, curr_rule, use_latex)
            curr_rule.pop()


    def postprocess_rules(self, curr_rule, use_latex):
        """ 
        Simplify numerical rules
        - Remove redundancy x1>3 and x1>5 becomes x1>5
        - Intervals x1>3 and x1<5 becomes 3<x1<5
        """
        
        curr_rule_copy = deepcopy(curr_rule)
        separators = [SYMBOLS[use_latex]["leq"], SYMBOLS[use_latex]["up"]]
        select_rules_0 = [rule for rule in curr_rule_copy if separators[0] in rule]
        splits_0 = [rule.split(separators[0])+[0] for rule in select_rules_0]
        select_rules_1 = [rule for rule in curr_rule_copy if separators[1] in rule]
        splits_1 = [rule.split(separators[1])+[1] for rule in select_rules_1]
        # There are splits
        if splits_0 or splits_1:
            splits = np.array(splits_0 + splits_1)
            select_features, inv, counts = np.unique(splits[:, 0], 
                                        return_inverse=True, return_counts=True)
            # There is redundancy
            if np.any(counts >= 2):
                # Iterate over redundant features
                for i in np.where(counts >= 2)[0]:
                    select_feature = select_features[i]
                    idxs = np.where(inv == i)[0]
                    # Remove the redundant rules
                    for idx in idxs:
                        curr_rule_copy.remove(select_feature+\
                                              separators[int(splits[idx, 2])]+\
                                              splits[idx, 1])
                    # Sort the rules in ascending order of threshold
                    argsort = idxs[np.argsort(splits[idxs, 1].astype(float))]
                    thresholds = splits[argsort, 1]
                    directions = splits[argsort, 2]
                    threshold_left = None
                    threshold_right = None
                    # We go from left to right and define the rule
                    for threshold, direction in zip(thresholds, directions):
                        # We stop at the first leq
                        if direction == '0':
                            threshold_left = threshold
                            break
                        if direction == '1':
                            threshold_right = threshold
                    # print the new rule
                    if threshold_right is None:
                        new_rule = select_feature+SYMBOLS[use_latex]["leq"]+threshold_left
                    elif threshold_left is None:
                        new_rule = select_feature+SYMBOLS[use_latex]["up"]+threshold_right
                    else:
                        new_rule = threshold_right + SYMBOLS[use_latex]["low"] +\
                            select_feature + SYMBOLS[use_latex]["leq"] + threshold_left
                    # Add the new rule
                    curr_rule_copy.append(new_rule)

        return curr_rule_copy


class CoE_Tree(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, decomposition):
        self.X = X
        self.N, self.D = X.shape
        self.H_add = get_h_add(decomposition)
        self.h = decomposition[()]
        self.loa_factor = 1 / self.h.var() # To have an loa [0, 1]
        self.n_groups = 0
        loa = np.mean((self.h - self.H_add.mean(1))**2)
        # Start recursive tree growth
        self.root, self.final_objective, self.n_groups = \
                self._tree_builder(np.arange(self.N), depth=0, loa=loa)
        self.final_loa = self.final_objective - self.alpha * self.n_groups
        return self


    def get_objective_for_splits(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], []
        
        # Otherwise we optimize the objective

        loa_left = np.zeros(len(splits))
        loa_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        h = self.h[instances_idx]
        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            loa_left[i]  = np.sum((h[x_i <= split] - self.H_add[left, left.T].mean(-1))**2)
            loa_right[i] = np.sum((h[x_i > split]  - self.H_add[right, right.T].mean(-1))**2)
        
        return splits[to_keep], loa_left[to_keep], loa_right[to_keep]



class PDP_PFI_Tree(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, decomposition):
        self.X = X
        self.N, self.D = X.shape
        self.h = decomposition[()]
        keys = decomposition.keys()
        additive_keys = [key for key in keys if len(key)==1]
        assert np.shape(decomposition[additive_keys[0]]) == (self.N, self.N), "Anchored decompositions must be provided"
        self.H = np.zeros((self.N, self.N, len(additive_keys)))
        # Additive terms
        for i, key in enumerate(additive_keys):
            self.H[..., i] = decomposition[key]
        
        self.loa_factor = 1 / self.h.var() # To have an loa 0-100%
        loa = np.mean(np.sum((self.H.mean(0) + self.H.mean(1))**2, axis=-1))
        # Start recursive tree growth
        self.root, self.final_objective, self.n_groups = \
                self._tree_builder(np.arange(self.N), depth=0, loa=loa)
        self.final_loa = self.final_objective - self.alpha * self.n_groups
        return self


    def get_objective_for_splits(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], []
        
        # Otherwise we optimize the objective
        loa_left = np.zeros(len(splits))
        loa_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            H_left = self.H[left, left.T]
            H_right = self.H[right, right.T]
            loa_left[i] = np.sum((H_left.mean(0) + H_left.mean(1))**2)
            loa_right[i] = np.sum((H_right.mean(0) + H_right.mean(1))**2)
        
        return splits[to_keep], loa_left[to_keep], loa_right[to_keep]



class PDP_SHAP_Tree(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, decomposition, Phi):
        self.X = X
        self.N, self.D = X.shape
        self.h = decomposition[()]
        # Construct the H tensor
        keys = decomposition.keys()
        additive_keys = [key for key in keys if len(key)==1]
        assert np.shape(decomposition[additive_keys[0]]) == (self.N, self.N), "Anchored decompositions must be provided"
        H = np.zeros((self.N, self.N, len(additive_keys)))
        # Additive terms
        for i, key in enumerate(additive_keys):
            H[..., i] = decomposition[key]
        # Compare with Phi tensor
        assert H.shape == Phi.shape
        self.Delta = Phi - H
        
        self.loa_factor = 1 / self.h.var() # To have an loa 0-100%
        loa = np.mean(np.sum(self.Delta.mean(1)**2, axis=-1))
        # Start recursive tree growth
        self.root, self.final_objective, self.n_groups = \
                self._tree_builder(np.arange(self.N), depth=0, loa=loa)
        self.final_loa = self.final_objective - self.alpha * self.n_groups
        return self


    def get_objective_for_splits(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], []
        
        # Otherwise we optimize the objective
        loa_left = np.zeros(len(splits))
        loa_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            Delta_left = self.Delta[left, left.T]
            Delta_right = self.Delta[right, right.T]
            loa_left[i] = np.sum(Delta_left.mean(1)**2)
            loa_right[i] = np.sum(Delta_right.mean(1)**2)
        
        return splits[to_keep], loa_left[to_keep], loa_right[to_keep]



class GADGET_PDP(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, X, decomposition):
        self.X = X
        self.N, self.D = X.shape
        self.h = decomposition[()]
        keys = decomposition.keys()
        additive_keys = [key for key in keys if len(key)==1]
        assert np.shape(decomposition[additive_keys[0]]) == (self.N, self.N), "Anchored decompositions must be provided"
        self.R = np.zeros((self.N, self.N, len(additive_keys)))
        # Additive terms
        for i, key in enumerate(additive_keys):
            self.R[..., i] = decomposition[key] + self.h
        
        self.loa_factor = 1 / self.h.var() # To have an loa 0-100%
        loa = np.mean(np.sum((self.R - self.R.mean(axis=0, keepdims=True) - 
                                self.R.mean(axis=1, keepdims=True) +
                                self.R.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))**2, 
                                axis=-1))
        # Start recursive tree growth
        self.root, self.final_objective, self.n_groups = \
                self._tree_builder(np.arange(self.N), depth=0, loa=loa)
        self.final_loa = self.final_objective - self.alpha * self.n_groups
        return self


    def get_objective_for_splits(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], []
        
        # Otherwise we optimize the objective
        loa_left = np.zeros(len(splits))
        loa_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            R_left = self.R[left, left.T]
            R_right = self.R[right, right.T]
            errors_left = (R_left - R_left.mean(axis=0, keepdims=True) - 
                            R_left.mean(axis=1, keepdims=True) +
                            R_left.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))**2
            loa_left[i] = errors_left.sum(-1).mean(-1).sum()
            errors_right = (R_right - R_right.mean(axis=0, keepdims=True) - 
                            R_right.mean(axis=1, keepdims=True) +
                            R_right.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))**2
            loa_right[i] = errors_right.sum(-1).mean(-1).sum()
        
        return splits[to_keep], loa_left[to_keep], loa_right[to_keep]



class CART(FDTree):
    """ 
    Classic CART that minimizes the Squared error 
    `sum_leaf sum_{x^(i)\in leaf} ( h(x^(i)) - v_leaf ) ^ 2`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, target):
        self.X = X
        self.N, self.D = X.shape
        self.target = target
        self.loa_factor = 1 / self.target.var() # To have an loss [0, 1]
        loa = self.target.var()
        # Start recursive tree growth
        self.root, self.final_objective, self.n_groups = \
                self._tree_builder(np.arange(self.N), depth=0, loa=loa)
        self.final_loa = self.final_objective - self.alpha * self.n_groups
        return self


    def get_objective_for_splits(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], [], [], []
        
        # Otherwise we optimize the objective
        objective_left = np.zeros(len(splits))
        objective_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split]
            right = instances_idx[x_i > split]
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            objective_left[i] = np.sum((self.target[left] - self.target[left].mean())**2)
            objective_right[i] = np.sum((self.target[right] - self.target[right].mean())**2)
        
        return splits[to_keep], objective_left[to_keep], objective_right[to_keep]


# class RandomTree(L2CoETree):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


#     def get_best_split(self, loa, curr_node, instances_idx):
#         splits = []
#         while len(splits) == 0:
#             best_feature_split = np.random.choice(range(self.D))
#             splits, N_left, N_right, loa_left, loa_right = \
#                                         self.get_split(instances_idx, best_feature_split)
#         # Chose a random split
#         idx = np.random.choice(range(max(1, len(splits)-1)))
#         # Otherwise search for the best split
#         best_split = splits[idx]
#         best_obj = (loa_right[idx]+loa_left[idx]) / len(instances_idx)
#         best_obj_left = loa_left[idx] / N_left[idx]
#         best_obj_right = loa_right[idx] / N_right[idx]

#         return best_feature_split, best_split, best_obj, best_obj_left, best_obj_right



@dataclass
class Partition:
    type: str = "coe"  # Type of partitionning "fd-tree" "random"
    save_losses : bool = True, # Save the tree locally
    alpha : float = 0.05 # Regularization of the FDTree
    min_samples_leaf : int = 30 # Minimum number of samples per leaf of the FDTree
    max_depth : int = 1 # Maximum depth of the FDTree
    branching_per_node : int = 1 # Number of splits to consider at each node (1 implies greedy)


PARTITION_CLASSES = {
    "coe": CoE_Tree,
    "pdp-pfi": PDP_PFI_Tree,
    "pdp-shap": PDP_SHAP_Tree,
    "gadget-pdp" : GADGET_PDP,
    "cart" : CART,
    # "random" : RandomTree,
}