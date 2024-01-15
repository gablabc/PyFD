""" 
Feature objects to represent feature of various types as well as operations on features such as selection and grouping
"""

import numpy as np
from copy import deepcopy

# Mappers take feature values and assing them a high level representation
# e.g. numerical -> (low, medium, high), categorical 3 -> "Married" etc.
# They are primarily used when visualising local attributions, where the 
# exact feature values may lack context.


# Boolean feature
class bool_value_mapper(object):
    """ Organize feature values as 1->true or 0->false """

    def __init__(self):
        self.values = ["False", "True"]

    # map 0->False  1->True
    def __call__(self, x):
        return self.values[round(x)]


# Ordinal/Nominal encoding of categorical features
class cat_value_mapper(object):
    """ Organize categorical features  int_value->'string_value' """

    def __init__(self, categories_in_order):
        self.cats = categories_in_order

    # x takes values 0, 1, 2 ,3  return the category
    def __call__(self, x):
        return self.cats[round(x)]


# Numerical features x in [xmin, xmax]
class numerical_value_mapper(object):
    """ Organize feature values in quantiles  value->{low, medium, high} """

    def __init__(self, num_feature_values):
        self.quantiles = np.quantile(num_feature_values, [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        bin_index = np.digitize(x, self.quantiles) - 1
        if bin_index == -1:
            return "very small (out)"
        elif bin_index == 5:
            return "very large (out)"
        else:
            return self.quantifiers[bin_index]


# Numerical features but with lots of zeros x in {0} U [xmin, xmax]
class sparse_numerical_value_mapper(object):
    """ Organize feature values in quantiles but treat 0-values differently """

    def __init__(self, num_feature_values):
        idx = np.where(num_feature_values != 0)[0]
        self.quantiles = np.quantile(
            num_feature_values[idx], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x == 0:
            return int(x)
        else:
            bin_index = np.digitize(x, self.quantiles) - 1
            if bin_index == -1:
                return "very small"
            elif bin_index == 5:
                return "very large (out)"
            else:
                return self.quantifiers[bin_index]



class Features(object):
    """ Abstraction of the concept of a feature """

    def __init__(self, X, feature_names, feature_types):
        """
        Parameters
        ----------
        X : (N, d) numpy array
            Dataset
        features_names : List(str)
            Names of the features for each column
        feature_types : List(str)
            A list indicating the type of each feature
            - `'num'` for numerical features
            - `'sparse_num'` for numerical features with many zeros
            - `'bool'` for True/False features
            - `'num_int'` for integer features
            - `('nominal', ['cat0', 'cat1'])` or `('ordinal', ['cat0', 'cat1'])` for categorical features
        """
        self.d = X.shape[1]
        assert self.d == len(feature_names), "feature_names must be of length d"
        assert self.d == len(feature_types), "feature_types must be of length d"
        # Map each feature to its group index
        self.Imap = np.arange(self.d).astype(int)
        # Map each group index to its features
        self.Imap_inv = [[i] for i in range(self.d)]
        self.names_ = feature_names

        self.types = []
        # Nominal categorical features that will need to be encoded
        self.nominal = []
        # map feature values to interpretable text
        self.maps_ = []
        for i, feature_type in enumerate(feature_types):
            # If its a list then the feature is categorical
            if type(feature_type) == list:
                self.types.append(feature_type[0]) # ordinal or nominal
                self.maps_.append(cat_value_mapper(feature_type[1:]))
                if feature_type[0] == "nominal":
                    self.nominal.append(i)
            else:   
                self.types.append(feature_type)
                if feature_type == "num":
                    self.maps_.append(numerical_value_mapper(X[:, i]))
                    
                elif feature_type == "sparse_num":
                    self.maps_.append(sparse_numerical_value_mapper(X[:, i]))
                    
                elif feature_type == "bool":
                    self.maps_.append(bool_value_mapper())
                    
                elif feature_type == "num_int":
                    self.maps_.append(lambda x: round(x))
                elif feature_type == "percent":
                    self.maps_.append(lambda x: f"{100*x:.0f}%")
                else:
                    raise ValueError("Wrong feature type")
                    
        # ordinal features are naturally represented with numbers
        self.ordinal = list( set(range(len(feature_types))) - set(self.nominal) )
    
    def print_value(self, x):
        """ Map values of x into interpretable text """
        print_res = []
        for i in range(len(self.Imap_inv)):
            if len(self.Imap_inv[i]) == 1:
                # feature=value
                idx = self.Imap_inv[i][0]
                print_res.append(f"{self.names_[idx]}={self.maps_[idx](x[idx])}")
            else:
                # for a group feature1=value1:feature2=value2
                print_res.append( ":".join([f"{self.names_[idx]}={self.maps_[idx](x[idx])}"
                                                                for idx in self.Imap_inv[i]]) )
        return print_res
    
    def print_names(self):
        """ Get the feature names """
        print_res = []
        for i in range(len(self.Imap_inv)):
            if len(self.Imap_inv[i]) == 1:
                # feature_name
                idx = self.Imap_inv[i][0]
                print_res.append(self.names_[idx])
            else:
                # for a group feature1_name:feature2_name
                print_res.append( ":".join([self.names_[idx] for idx in self.Imap_inv[i]]) )
        return print_res

    def __len__(self):
        return self.d
    
    # def select(self, i_range):
    #     """ Return a copy using only a subset of features """
    #     feature_copy = deepcopy(self)
    #     feature_copy.names = [feature_copy.names[i] for i in i_range]
    #     feature_copy.types = [feature_copy.types[i] for i in i_range]
    #     feature_copy.maps  = [feature_copy.maps[i] for i in i_range]
    #     # TODO handle nominal and ordinal
    #     feature_copy.nominal = []
    #     feature_copy.ordinal = []
    #     return feature_copy
    
    def group(self, feature_groups):
        """ 
        Put the select feature into groups. This will update the 
        Imap and Imap_inv attributes of the class

        Parameters
        ----------
        feature_groups : List(List(int))
            A List containing the groups to form `[[0, 1, 2], [3, 4, 5]]`
            will group the features 0, 1, 2 together for example.
        
        Returns
        -------
        feature_copy : Features
            A copy of the Feature instance with grouped features
            and updated I_map and I_map_inv
        """
        assert type(feature_groups) in (tuple, list)
        assert type(feature_groups[0]) in (tuple, list)

        feature_copy = deepcopy(self)
        # Update Imap_inv and feature_names
        for feature_group in feature_groups:
            feature_copy.Imap_inv.append(feature_group)
            for feature in feature_group:
                feature_copy.Imap_inv.remove([feature])
        # Update I_map
        counter = 0
        for feature_group in feature_copy.Imap_inv:
            for feature in feature_group:
                feature_copy.Imap[feature] = counter
            counter += 1

        # TODO Rearrange types ??
        feature_copy.types = []
        feature_copy.nominal = []
        feature_copy.ordinal = []
        return feature_copy



# Debugging
if __name__ == "__main__":
    features = Features(np.random.uniform(-1, 1, size=(1000, 5)),
                        feature_names=[f"x{i}" for i in range(5)],
                        feature_types=['num'] * 5)
    print(features.names_)
    print(features.print_names())
    print(features.print_value(np.zeros(5)))
    print(features.print_value(2 * np.ones(5)))
    print(features.Imap)
    print(features.Imap_inv, "\n")

    # Group some features
    grouped_features = features.group([[1, 3]])
    print(features.names_)
    print(grouped_features.print_names())
    print(grouped_features.print_value(np.zeros(5)))
    print(grouped_features.print_value(2 * np.ones(5)))
    print(grouped_features.Imap)
    print(grouped_features.Imap_inv)