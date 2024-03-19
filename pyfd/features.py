""" 
Feature objects to represent feature of various types as well as operations on features such as selection and grouping
"""

import numpy as np
from copy import deepcopy

# A feature is an object that has a `name` attribute, a `type` attribute, and a callable
# that takes x and return a string representation of the feature. These are a high level representation
# e.g. numerical -> (low, medium, high), categorical 3 -> "Married" etc.
# These representation are primarily used when visualising local attributions, where the 
# exact feature values may lack context.


# Boolean feature
class bool_feature(object):
    """ Organize feature values as 1->true or 0->false """

    def __init__(self, name):
        self.name = name
        self.type = "bool"
        self.values = ["False", "True"]
        self.card = 2

    # map 0->False  1->True
    def __call__(self, x):
        return f"{self.name}={self.values[round(x)]}"


# Ordinal/Nominal encoding of categorical features
class cat_feature(object):
    """ Organize categorical features  int_value->'string_value' """

    def __init__(self, name, type, categories_in_order):
        self.name = name
        self.type = type # ordinal or nominal
        self.cats = categories_in_order
        self.card = len(self.cats)

    # x takes values 0, 1, 2 ,3  return the category
    def __call__(self, x):
        return f"{self.name}={self.cats[round(x)]}"


# Numerical features x in [xmin, xmax]
class numerical_feature(object):
    """ Organize feature values in quantiles  value->{low, medium, high} """

    def __init__(self, name, num_feature_values):
        self.name = name
        self.type = "num"
        self.quantiles = np.quantile(num_feature_values, [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]
        self.card = np.inf

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        bin_index = np.digitize(x, self.quantiles) - 1
        if bin_index == -1:
            str_value = "very small (out)"
        elif bin_index == 5:
            str_value = "very large (out)"
        else:
            str_value = self.quantifiers[bin_index]
        return self.name + "=" + str_value


# Numerical features but with lots of zeros x in {0} U [xmin, xmax]
class sparse_numerical_feature(object):
    """ Organize feature values in quantiles but treat 0-values differently """

    def __init__(self, name, num_feature_values):
        self.name = name
        self.type = "sparse_num"
        idx = np.where(num_feature_values != 0)[0]
        self.quantiles = np.quantile(
            num_feature_values[idx], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]
        self.card = np.inf

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x == 0:
            str_value = str(int(x))
        else:
            bin_index = np.digitize(x, self.quantiles) - 1
            if bin_index == -1:
                str_value = "very small"
            elif bin_index == 5:
                str_value = "very large (out)"
            else:
                str_value = self.quantifiers[bin_index]
        return self.name + "=" + str_value


# Integer feature x=0, 1, 2 ,3, ...
class integer_feature(object):
    """ Feature that takes intefer values 0, 1, 2, 3, ... """

    def __init__(self, name, values):
        self.name = name
        self.type = "num_int"
        self.min = values.min()
        self.max = values.max()
        self.card = int(self.max - self.min + 1)

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x < self.min:
            str_value = "very small (out)"
        elif x > self.max:
            str_value = "very large (out)"
        else:
            str_value = str(round(x))
        return self.name + "=" + str_value
    

# Feature that takes [0, 1] values which represent percentage e.g. 0.5->50%
class percent_feature(object):
    """ Feature that takes [0, 1] values which represent percentage e.g. 0.5->50% """

    def __init__(self, name):
        self.name = name
        self.type = "percent"
        self.card = np.inf

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x < 0 or x > 1:
            str_value = "out of scope"
        else:
            str_value = f"{100*x:.0f}%"
        return self.name + "=" + str_value

class combined_feature(object):
    """ Plot the values of joined features e.g. feature1=dog:feature2=42 """

    def __init__(self, feature_objs):
        self.names = [obj.name for obj in feature_objs]
        self.name = ":".join(self.names)
        self.type = ":".join([obj.type for obj in feature_objs])
        self.card = ":".join([str(obj.card) for obj in feature_objs])
        self.feature_objs = feature_objs
        self.n_features = len(feature_objs)

    # map feature1=value1:feature2=value2
    def __call__(self, x):
        assert len(x) == self.n_features
        return ":".join([self.feature_objs[i](x[i]) for i in range(self.n_features)])


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

        # Nominal categorical features that will need to be encoded
        self.nominal = []
        # A list of feature objects
        self.feature_objs = []
        for i, feature_type in enumerate(feature_types):
            # If its a list then the feature is categorical
            if type(feature_type) in (list, tuple):
                self.feature_objs.append(cat_feature(feature_names[i],
                                                     feature_type[0], 
                                                     feature_type[1:]))
                if feature_type[0] == "nominal":
                    self.nominal.append(i)
            else:   
                if feature_type == "num":
                    self.feature_objs.append(numerical_feature(feature_names[i], X[:, i]))
                    
                elif feature_type == "sparse_num":
                    self.feature_objs.append(sparse_numerical_feature(feature_names[i], X[:, i]))
                    
                elif feature_type == "bool":
                    self.feature_objs.append(bool_feature(feature_names[i]))
                    
                elif feature_type == "num_int":
                    self.feature_objs.append(integer_feature(feature_names[i], X[:, i]))

                elif feature_type == "percent":
                    self.feature_objs.append(percent_feature(feature_names[i]))

                else:
                    raise ValueError("Wrong feature type")
                    
        # ordinal features are naturally represented with numbers
        self.ordinal = list( set(range(len(feature_types))) - set(self.nominal) )
    
    def print_value(self, x):
        """ Map values of x into interpretable text """
        print_res = []
        for i in range(len(self.Imap_inv)):
            if len(self.Imap_inv[i]) == 1:
                # Returns a int/float
                x_i = x[self.Imap_inv[i][0]]
            else:
                # Returns an array
                x_i = x[self.Imap_inv[i]]
            print_res.append( self.feature_objs[i](x_i) )
        return print_res
    
    def print_names(self):
        return [obj.name for obj in self.feature_objs]

    def print_types(self):
        return [obj.type for obj in self.feature_objs]
    
    def summary(self):
        free_space = [20, 20, 10, 20]
        print_res =  "|        Name        |        Type        |   Card   |      I^-1({i})     |\n"
        print_res += "---------------------------------------------------------------------------\n"
        for i in range(len(self.Imap_inv)):
            print_res += "|"
            obj = self.feature_objs[i]
            properties = [obj.name, obj.type, obj.card, self.Imap_inv[i]]
            for j in range(4):
                property = str(properties[j])
                if len(property) < free_space[j]:
                    space = free_space[j] - len(property) - 1
                else:
                    space = 0
                    property = property[:free_space]
                print_res += " " + property + space * " " + "|"
            print_res += "\n"
        print_res += "---------------------------------------------------------------------------\n"
        return print_res

    def __len__(self):
        return len(self.Imap_inv)
    
    def select(self, feature_idxs):
        """ 
        Select the listed features. This will update the 
        `Imap_inv` and `feature_objs` attributes of the class

        Parameters
        ----------
        feature_idxs : List(int)
            A List containing the index of the features to select.
        
        Returns
        -------
        feature_copy : Features
            A copy of the Feature instance with selected features
            and updated `Imap_inv` and `feature_objs`
        """

        assert type(feature_idxs) in (tuple, list)
        feature_copy = deepcopy(self)
        feature_copy.Imap_inv = []
        feature_copy.feature_objs = []
        feature_copy.nominal = []
        feature_copy.ordinal = []
        # Imap does not make sense anymore since we not all data columns
        # are considered
        self.Imap = None

        for idx in feature_idxs:
            feature_copy.Imap_inv.append( self.Imap_inv[idx] )
            feature_copy.feature_objs.append( self.feature_objs[idx] )
        
        return feature_copy
    

    def remove(self, feature_idxs):
        """ 
        Remove the listed features. This will update the 
        `Imap_inv` and `feature_objs` attributes of the class

        Parameters
        ----------
        feature_idxs : List(int)
            A List containing the index of the features to remove.
        
        Returns
        -------
        feature_copy : Features
            A copy of the Feature instance with removec features
            and updated `Imap_inv` and `feature_objs`
        """
        return self.select( [i for i in range(len(self.Imap_inv)) if not i in feature_idxs] )


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
        # Update I_map, and feature_objs
        counter = 0
        feature_copy.feature_objs = []
        for feature_group in feature_copy.Imap_inv:
            # Update Imap
            for idx in feature_group:
                feature_copy.Imap[idx] = counter
            # Update types
            if len(feature_group) == 1:
                feature_copy.feature_objs.append( self.feature_objs[feature_group[0]] )
            else:
                feature_copy.feature_objs.append( combined_feature([self.feature_objs[idx] for idx in feature_group]) )
            counter += 1

        feature_copy.nominal = []
        feature_copy.ordinal = []
        return feature_copy



# Debugging
if __name__ == "__main__":
    X = np.column_stack((np.random.uniform(-1, 1, size=(1000,)),
                          np.random.randint(0, 2, size=(1000,)),
                          np.random.uniform(-1, 1, size=(1000,)),
                          np.random.uniform(-1, 1, size=(1000,)),
                          np.random.randint(0, 2, size=(1000,))
    ))
    features = Features(X, feature_names=[f"x{i}" for i in range(5)],
                        feature_types=["num", "num_int", "num", "num", ("nominal", "cat", "dog")]
                        )
    print(features.summary())
    print(features.print_names())
    print(features.print_types())
    print(features.print_value(np.zeros(5)))
    print(features.print_value(np.ones(5)))
    print(features.Imap, "\n\n")

    # Group some features
    grouped_features = features.group([[1, 3]])
    print(grouped_features.summary())
    print(grouped_features.print_names())
    print(grouped_features.print_types())
    print(grouped_features.print_value(np.zeros(5)))
    print(grouped_features.print_value(np.ones(5)))
    print(grouped_features.Imap, "\n\n")

    # Keep some features
    keep_features = features.select([0, 2, 4])
    print(keep_features.summary())
    print(keep_features.print_names())
    print(keep_features.print_types())
    print(keep_features.print_value(np.zeros(5)))
    print(keep_features.print_value(np.ones(5)))
    print(keep_features.Imap, "\n\n")

    # Remove some features
    remove_features = features.remove([2])
    print(remove_features.summary())
    print(remove_features.print_names())
    print(remove_features.print_types())
    print(remove_features.print_value(np.zeros(5)))
    print(remove_features.print_value(np.ones(5)))
    print(remove_features.Imap, "\n\n")