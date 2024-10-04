""" Assert that the Decomposition Functions are correct """

import pytest
import numpy as np

from pyfd.features import *


def setup_Features():
    X = np.column_stack((np.random.uniform(-1, 1, size=(1000,)),
                          np.random.randint(0, 2, size=(1000,)),
                          np.random.uniform(-1, 1, size=(1000,)),
                          np.random.uniform(-1, 1, size=(1000,)),
                          np.random.randint(0, 2, size=(1000,))
    ))
    features = Features(X, names=[f"x{i}" for i in range(5)],
                        types=["num", "num_int", "num", "num", ("nominal", "cat", "dog")])
    return features


###################################################################################################
#                                                  Tests                                          #
###################################################################################################


def test_features_objs():
    """ Test the feature objects """
    
    # test boolean feature
    feature = bool_feature("Workinday")
    assert feature(0) == "notWorkinday"
    assert feature(1) == "Workinday"
    with pytest.raises(Exception):
        print(features(-1))
        print(features(2))
           
    # test percent feature
    feature = percent_feature("Effort")
    assert feature(0.2) == "Effort=20%"
    assert feature(0.55) == "Effort=55%"
    assert feature(0.999) == "Effort=100%"
    assert feature(-1) == "Effort=out of scope"
    assert feature(1.2) == "Effort=out of scope"

    # test integer feature
    feature = integer_feature("Age", [25, 70])
    assert feature(10) == "Age=very small (out)"
    assert feature(50) == "Age=50"
    assert feature(44) == "Age=44"
    assert feature(89) == "Age=very large (out)"

    # test numerical feature
    feature = numerical_feature("Salary", np.random.uniform(10, 100, size=(1000,)))
    quantiles = feature.quantiles
    assert feature(5) == "Salary=very small (out)"
    assert feature(quantiles[0]) == "Salary=very small"
    assert feature(np.mean(quantiles[[0, 1]])) == "Salary=very small"
    assert feature(np.mean(quantiles[[1, 2]])) == "Salary=small"
    assert feature(np.mean(quantiles[[2, 3]])) == "Salary=medium"
    assert feature(np.mean(quantiles[[3, 4]])) == "Salary=large"
    assert feature(np.mean(quantiles[[4, 5]])) == "Salary=very large"
    assert feature(quantiles[-1]) == "Salary=very large (out)"
    assert feature(200) == "Salary=very large (out)"

    # test categorical feature
    feature = cat_feature("Animal", "nominal", ["Cats", "Dogs", "Hamsters"])
    assert feature(0) == "Animal=Cats"
    assert feature(1) == "Animal=Dogs"
    assert feature(2) == "Animal=Hamsters"
    with pytest.raises(Exception):
        print(features(4))



def test_group_feature_objs():
    """ Test grouping feature objects """

    # test two features 
    feature_1 = bool_feature("Workingday")
    feature_2 = cat_feature("Animal", "nominal", ["Cats", "Dogs", "Hamsters"])
    grouped_feature = combined_feature([feature_1, feature_2])
    assert grouped_feature.name == "Workingday:Animal"
    assert grouped_feature.type == "bool:nominal"
    assert grouped_feature.card == "2:3"
    assert grouped_feature([0, 2]) == "notWorkingday:Animal=Hamsters"
    assert grouped_feature([1, 1]) == "Workingday:Animal=Dogs"

    # test feature and group
    feature_3 = percent_feature("Effort")
    grouped_feature_2 = combined_feature([grouped_feature, feature_3])
    assert grouped_feature_2.name == "Workingday:Animal:Effort"
    assert grouped_feature_2.type == "bool:nominal:percent"
    assert grouped_feature_2.card == "2:3:inf"
    assert grouped_feature_2([0, 2, 0.55]) == "notWorkingday:Animal=Hamsters:Effort=55%"
    assert grouped_feature_2([1, 1, 0.13]) == "Workingday:Animal=Dogs:Effort=13%"

    # test group and group
    feature_4 = numerical_feature("Salary", np.random.uniform(10, 100, size=(1000,)))
    grouped_feature_2 = combined_feature([feature_3, feature_4])
    grouped_feature_3 = combined_feature([grouped_feature, grouped_feature_2])
    assert grouped_feature_3.name == "Workingday:Animal:Effort:Salary"
    assert grouped_feature_3.type == "bool:nominal:percent:num"
    assert grouped_feature_3.card == "2:3:inf:inf"
    assert grouped_feature_3([0, 2, 0.55, 0]) == "notWorkingday:Animal=Hamsters:Effort=55%:Salary=very small (out)"
    
    
def test_features_select():
    """ Test the selection of a subset of features """
    features  = setup_Features()
    select_features = features.select([1, 2, 4])
    assert len(select_features) == 3
    assert select_features.names() == ["x1", "x2", "x4"]
    assert select_features.types() == ["num_int", "num", "nominal"]
    assert select_features.Imap_inv == [[1], [2], [4]]


def test_features_remove():
    """ Test the removal of a subset of features """
    features  = setup_Features()
    select_features = features.remove([1, 2, 4])
    assert len(select_features) == 2
    assert select_features.names() == ["x0", "x3"]
    assert select_features.types() == ["num", "num"]
    assert select_features.Imap_inv == [[0], [3]]


def test_features_group():
    """ Test the grouping of a subset of features """
    features  = setup_Features()
    select_features = features.group([[1, 4]])
    assert len(select_features) == 4
    assert select_features.names() == ["x0", "x2", "x3", "x1:x4"]
    assert select_features.types() == ["num", "num", "num", "num_int:nominal"]
    assert select_features.Imap_inv == [[0], [2], [3], [1, 4]]


def test_features_combinations():
    """ Test the removal and grouping of a subset of features """
    features  = setup_Features()
    new_features = features.group([[1, 4]]).remove([2]).group([[0, 1]])
    new_features.summary()
    assert len(new_features) == 2
    assert new_features.names() == ["x1:x4", "x0:x2"]
    assert new_features.types() == ["num_int:nominal", "num:num"]
    assert new_features.Imap_inv == [[1, 4], [0, 2]]

    new_features = features.group([[1, 4]]).group([[1, 3]])
    new_features.summary()
    assert len(new_features) == 3
    assert new_features.names() == ["x0", "x3", "x2:x1:x4"]
    assert new_features.types() == ["num", "num", "num:num_int:nominal"]
    assert new_features.Imap_inv == [[0], [3], [2, 1, 4]]


if __name__ == "__main__":
    test_features_combinations()
