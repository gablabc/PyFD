""" Build-in functions for loading XAI datasets """

import pandas as pd
import os
import numpy as np

# Sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from .features import Features


class TargetEncoder(BaseEstimator, TransformerMixin):
    """ 
    Encode nominal features in terms of the frequency of the target
    conditioned on each category.
    """
    def __init__(self):
        pass

    def fit(self, df):
        """ 
        Fit the estimator
        
        Parameters
        ----------
        df : (N, d+1) dataframe
            A dataframe of nominal features whose last column is the label
        
        Returns
        -------
        self
        """
        ncols = df.shape[1] - 1
        target = df.columns[-1]
        self.categories_ = [0] * ncols
        for i, feature in enumerate(df.columns[:-1]):
            frequencies = []
            self.categories_[i] = np.unique(df[feature])
            for category in self.categories_[i]:
                frequencies.append(df[df[feature]==category][target].mean())
            argsort = np.argsort(frequencies)
            self.categories_[i] = self.categories_[i][argsort]
        return self


    def transform(self, df):
        """ 
        Encode the provided data
        
        Parameters
        ----------
        df : (N, d+1) dataframe
            A dataframe of nominal features whose last column is the label
        
        Returns
        -------
        X : (N, d) np.ndarray
            Nominal features encoded as ordinal.
        """
        ncols = df.shape[1] - 1
        X = np.zeros((df.shape[0], ncols))
        for i, feature in enumerate(df.columns[:-1]):
            for j, category in enumerate(self.categories_[i]):
                idx = np.where(df[feature]==category)[0]
                X[idx, i] = j
        return X




def get_data_compas():
    """ 
    Load the COMPAS dataset for recidivism prediction 
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """
    
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                    "datasets", "COMPAS", "compas-scores-two-years.csv"))
    # Same preprocessing as done by ProPublica but we also only keep Caucasians and Blacks
    keep = (df["days_b_screening_arrest"] <= 30) &\
        (df["days_b_screening_arrest"] >= -30) &\
        (df["score_text"] != "nan") &\
        ((df["race"] == "Caucasian") | (df["race"] == "African-American")) 
    df = df[keep]

    # Binarize some features
    df.loc[:, 'sex_Male'] = (df['sex'] == 'Male').astype(int)
    df.loc[:, 'race_Black'] = (df['race'] == "African-American").astype(int)
    df.loc[:, 'c_charge_degree_F'] = (df['c_charge_degree'] == 'F').astype(int)

    # Features to keep
    features = ['sex_Male', 'race_Black', 'c_charge_degree_F',
                'priors_count', 'age', 'juv_fel_count', 'juv_misd_count']
    X = df[features]

    # Rename some columns
    X = X.rename({"sex_Male" : "Sex", "race_Black" : "Race", "c_charge_degree_F" : "Charge", 
              "priors_count" : "Priors", "age" : "Age", "juv_fel_count" : "Juv_felonies", 
              "juv_misd_count" : "Juv_misds"})
    X = X.to_numpy().astype(np.float64)
    # New Features to keep
    feature_names = ['Sex', 'Race', 'Charge', 'Priors', 'Age', 'JuvFelonies', 'JuvMisds']

    # Target
    # y = df["decile_score"].to_numpy().astype(np.float64)
    y = df["two_year_recid"].astype(int)

    # Generate Features object
    feature_types = [
        ["ordinal", "Female", "Male"],
        ["ordinal", "White", "Black"],
        ["ordinal", "Misd", "Felony"],
        "num_int",
        "num_int",
        "num_int",
        "num_int"
    ]

    features = Features(X, feature_names, feature_types)

    return X, y, features




def get_data_bike():
    """ 
    Load the BikeSharing dataset for bike rental predictions 
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """
    
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "datasets", "Bike-Sharing/hour.csv")
    )
    df.drop(columns=["dteday", "casual", "registered", "instant"], inplace=True)

    # Remove correlated features
    df.drop(columns=["atemp", "season"], inplace=True)

    # Rescale temp to Celcius
    df["temp"] = 41 * df["temp"]

    # Month count starts at 0
    df["mnth"] -= 1

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)

    # Scale all features
    feature_names = list(df.columns[:-1])

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Generate Features object
    feature_types = [
        ["ordinal", "2011", "2012"],
        ["ordinal",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October","November", "December",
        ],
        "num_int",
        "bool",
        ["ordinal",
            "Sunday", "Monday", "Thuesday", "Wednesday", "Thursday",
            "Friday", "Saturday"],
        "bool",
        "num_int",
        "num",
        "num",
        "num",
    ]

    features = Features(X, feature_names, feature_types)

    return X, y, features




def get_data_adults(use_target_encoder=False):
    """ 
    Load the Adult-Income dataset for income predictions 
    
    Parameters
    ----------
    use_target_encoder : bool, default=False
        Encode nominal features using the TargetEncoder
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """
    # load train
    raw_data_1 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.data'), 
                                                     delimiter=', ', dtype=str)
    # load test
    raw_data_2 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.test'),
                                      delimiter=', ', dtype=str, skip_header=1)

    feature_names = ['age', 'workclass', 'fnlwgt', 'education',
                     'educational-num', 'marital-status', 'occupation', 
                     'relationship', 'race', 'gender', 'capital-gain', 
                     'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Shuffle train/test
    df = pd.DataFrame(np.vstack((raw_data_1, raw_data_2)), columns=feature_names)


    # For more details on how the below transformations 
    df = df.astype({"age": np.int64, "educational-num": np.int64, 
                    "hours-per-week": np.int64, "capital-gain": np.int64, 
                    "capital-loss": np.int64 })

    # Reduce number of categories
    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 
                                   'Never-worked': 'Other/Unknown'}})
    df = df.replace({'workclass': {'?': 'Other/Unknown'}})
    df = df.replace({'workclass': {'Federal-gov': 'Government', 
                                   'State-gov': 'Government', 'Local-gov':'Government'}})
    df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 
                                   'Self-emp-inc': 'Self-Employed'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 
                                    'Craft-repair': 'Blue-Collar',
                                    'Exec-managerial':'White-Collar',
                                    'Farming-fishing':'Blue-Collar',
                                    'Handlers-cleaners':'Blue-Collar',
                                    'Machine-op-inspct':'Blue-Collar',
                                    'Other-service':'Service',
                                    'Priv-house-serv':'Service',
                                    'Prof-specialty':'Professional',
                                    'Protective-serv':'Service',
                                    'Tech-support':'Service',
                                    'Transport-moving':'Blue-Collar',
                                    'Unknown':'Other/Unknown',
                                    'Armed-Forces':'Other/Unknown',
                                    '?':'Other/Unknown'}})

    df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 
                                        'Married-AF-spouse': 'Married', 
                                        'Married-spouse-absent':'Married',
                                        'Never-married':'Single'}})

    df = df.replace({'income': {'<=50K': 0, '<=50K.': 0,  '>50K': 1, '>50K.': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                   '11th':'School', '10th':'School', 
                                   '7th-8th':'School', '9th':'School',
                                   '12th':'School', '5th-6th':'School', 
                                   '1st-4th':'School', 'Preschool':'School'}})

    # Put numeric+ordinal before nominal and remove fnlwgt-country
    df = df[['age', 'educational-num', 'capital-gain', 'capital-loss',
             'hours-per-week', 'gender', 'workclass','education', 'marital-status', 
             'occupation', 'relationship', 'race', 'income']]
    df = shuffle(df, random_state=42)
    feature_names = df.columns[:-1]

    # Make a column transformer for ordinal encoder
    if use_target_encoder:
        encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), df.columns[:5]),
                       ('encoder', TargetEncoder(), df.columns[5:])
                      ])
        X = encoder.fit_transform(df)
        cat_type = "ordinal"
    else:
        encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), df.columns[:5]),
                       ('encoder', OrdinalEncoder(), df.columns[5:-1])
                      ])
        X = encoder.fit_transform(df.iloc[:, :-1])
        cat_type = "nominal"
    y = df["income"].to_numpy()
    
    # Generate Features object
    feature_types = ["num", "num", "sparse_num", "sparse_num", "num"] +\
        [([cat_type] + list(l)) for l in encoder.transformers_[1][1].categories_]
    features = Features(X, feature_names, feature_types)
    
    return X, y, features



def get_data_marketing(use_target_encoder=False):
    """ 
    Load the Marketing dataset for phone-call success predictions 
    
    Parameters
    ----------
    use_target_encoder : bool, default=False
        Encode nominal features using the TargetEncoder
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """

    # load train
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                    'datasets', 'marketing', 'marketing.csv'), delimiter=";")
    feature_names = df.columns[:-1]
    outcome = df.columns[-1]

    # Shuffle the dataset since it is ordered w.r.t time
    df = df.sample(frac=1, random_state=42)
    
    # Replace yes/no with 1/0
    binary_columns = ["default", "housing", "loan", outcome]
    for binary_column in binary_columns:
        df[binary_column] = (df[binary_column] == "yes").astype(int)

    # Months should be number jan=0 feb=1 etc
    months =["jan", "feb", "mar", "apr", "may", "jun", "jul", 
                     "aug", "sep", "oct", "nov", "dec"]
    df = df.replace(months, range(12))
    
    # Unknown -> ?
    df = df.replace("unknown", "?")

    # Categorical and numerical features
    cat_cols = [1, 2, 3, 8, 15, 16]
    num_cols = [0, 4, 5 ,6 ,7, 9, 10, 11, 12, 13, 14]
    feature_names = [feature_names[i] for i in num_cols] + \
                    [feature_names[i] for i in cat_cols[:-1]]

    # Make a column transformer for ordinal encoder
    if use_target_encoder:
        encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), num_cols),
                       ('encoder', TargetEncoder(), cat_cols)
                      ])
        X = encoder.fit_transform(df)
    else:
        encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), num_cols),
                       ('encoder', OrdinalEncoder(), cat_cols[:-1])
                      ])
        X = encoder.fit_transform(df.iloc[:, :-1])
    y = df[outcome].to_numpy()
    
    # Generate Features object
    feature_types = ["num_int", "bool", "num", "bool", "bool", "num_int"] + \
        [["ordinal"]+ months] + ["num"]*4 + \
        [(["ordinal"] + list(l)) for l in encoder.transformers_[1][1].categories_]
    features = Features(X, feature_names, feature_types)
    
    return X, y, features



def get_data_credit(use_target_encoder=False):
    """ 
    Load the Default-Credit dataset for loan default predictions 
    
    Parameters
    ----------
    use_target_encoder : bool, default=False
        Encode nominal features using the TargetEncoder
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """

    # load train
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                            'datasets', 'default_credit', 'default_credit.csv'))
    # Rename columns to make their name more interpretable
    df = df.rename(columns={"LIMIT_BAL": "Limit", "SEX" : "Gender", 
                    "EDUCATION"  :"Education", "MARRIAGE" : "Mariage", "AGE" : "Age",
                    "PAY_0" : "Delay-Sep", "PAY_2": "Delay-Aug", "PAY_3": "Delay-Jul", 
                    "PAY_4" : "Delay-Jun", "PAY_5" : "Delay-May", "PAY_6" : "Delay-Apr",
                    "BILL_AMT1" : "Bill-Sep", "BILL_AMT2" : "Bill-Aug", "BILL_AMT3" : "Bill-Jul",
                    "BILL_AMT4" : "Bill-Jun" , "BILL_AMT5" : "Bill-May", "BILL_AMT6" : "Bill-Apr",
                    "PAY_AMT1" : "Pay-Sep", "PAY_AMT2" : "Pay-Aug", "PAY_AMT3" : "Pay-Jul",
                    "PAY_AMT4" : "Pay-Jun" , "PAY_AMT5" : "Pay-May", "PAY_AMT6" : "Pay-Apr",
                    "DEFAULT_PAYEMENT" : "Default"})
    feature_names = df.columns[:-1]
    outcome = df.columns[-1]
    
    # Remove the 14 rows with Education=Nan
    df = df.dropna()

    # Replace with 1/0
    binary_columns = ["Delay-Sep", "Delay-Aug", "Delay-Jul", 
                      "Delay-Jun", "Delay-May", "Delay-Apr"]
    for binary_column in binary_columns:
        df[binary_column] = (df[binary_column] == "Pay_delay>=1").astype(int)
    
    # Categorical and numerical features
    cat_cols = [1, 2, 3, 23]
    num_cols = [0] + list(range(4, 23))
    feature_names = [feature_names[i] for i in num_cols] + \
                    [feature_names[i] for i in cat_cols[:-1]]

    # Make a column transformer for ordinal encoder
    if use_target_encoder:
        encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), num_cols),
                       ('encoder', TargetEncoder(), cat_cols)
                      ])
        X = encoder.fit_transform(df)
    else:
        encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), num_cols),
                       ('encoder', OrdinalEncoder(), cat_cols[:-1])
                      ])
        X = encoder.fit_transform(df.iloc[:, :-1])
    y = df[outcome].to_numpy()
    
    # Generate Features object
    feature_types = ["num", "num_int"] + ["bool"]*6 + ["sparse_num"]*12 +\
        [(["ordinal"] + list(l)) for l in encoder.transformers_[1][1].categories_]
    features = Features(X, feature_names, feature_types)
    
    return X, y, features



def get_data_kin8nm():
    """ 
    Load the Kin8nm dataset for robot-arm dynamics predictions 
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """

    # load train
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                    'datasets', 'kin8nm', 'dataset_2175_kin8nm.csv'), delimiter=",")
    feature_names = list(df.columns[:-1])
    outcome = df.columns[-1]
    
    X = df[feature_names].to_numpy()
    y = df[outcome].to_numpy()
    
    # Generate Features object
    feature_types = ["num"]*8
    features = Features(X, feature_names, feature_types)
    
    return X, y, features



def get_data_california_housing():
    """ 
    Load the California dataset for house pricing predictions 
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """

    data = fetch_california_housing()

    X, y, feature_names = data["data"], data["target"], data["feature_names"]

    # Remove outlier
    keep_bool = X[:, 5] < 1000
    X = X[keep_bool]
    y = y[keep_bool]
    del keep_bool

    # Take log of right-skewed features
    for i in [2, 3, 5]:
        X[:, i] = np.log10(X[:, i])
        feature_names[i] = f"log{feature_names[i]}"

    # # Add additionnal location feature
    # def closest_point(location):
    #     # Biggest cities in 1990
    #     # Los Angeles, San Francisco, San Diego, San Jose
    #     biggest_cities = [
    #         (34.052235, -118.243683),
    #         (37.773972, -122.431297),
    #         (32.715736, -117.161087),
    #         (37.352390, -121.953079),
    #     ]
    #     closest_location = None
    #     for city_x, city_y in biggest_cities:
    #         distance = ((city_x - location[0]) ** 2 + (city_y - location[1]) ** 2) ** (
    #             1 / 2
    #         )
    #         if closest_location is None:
    #             closest_location = distance
    #         elif distance < closest_location:
    #             closest_location = distance
    #     return closest_location

    # X = np.column_stack((X, [closest_point(x[-2:]) for x in X]))
    # feature_names.append('ClosestBigCityDist')

    # Generate Features object
    feature_types = ["num", "num", "num", "num", "num", "num", "num", "num"]
    
    features = Features(X, feature_names, feature_types)
    
    return X, y, features




def get_data_kaggle_housing(remove_correlations=False, submission=False):
    """ 
    Load the Kaggle-Houses dataset for house pricing predictions 
    
    Parameters
    ----------
    remove_correlations : bool, default=False
        Remove correlated features from the dataset

    subsmission : bool, default=False
        Load the test data (without labels) to submit on kaggle
    
    Returns
    -------
    X : (N, d) np.ndarray
        The input features
    y : (N,) np.ndarray
        The label
    features : pyfd.features.Features
        feature object
    """

    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv
    
    if submission:
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "datasets", "kaggle_houses", "test.csv"
            )
        )
    else:
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "datasets", "kaggle_houses", "train.csv"
            )
        )
    Id = df["Id"]

    # dropping categorical features
    df.drop(
        labels=[
            "Id", "MSSubClass", "MSZoning", "Street", "Alley", "LotShape",
            "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
            "Condition1", "Condition2", "BldgType", "HouseStyle", "MSZoning",
            "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
            "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
            "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC",
            "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType",
            "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
            "MiscFeature", "SaleType", "SaleCondition", "CentralAir","PavedDrive"
        ],
        axis=1,
        inplace=True,
    )

    #### Missing Data ####
    # Replace missing values by the median
    columns_with_nan = df.columns[np.where(df.isna().any())[0]]
    if columns_with_nan is not None:
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        df[columns_with_nan] = imp.fit_transform(df[columns_with_nan])

    #### Features which are mostly zero ####
    # Dropping LotFrontage because it is missing 259/1460 values 
    # which is a lot (GarageYrBlt: 81 and MasVnrArea: 8 is reasonable)
    df.drop(labels=["LotFrontage"], axis=1, inplace=True)
    # 1408 houses have MiscVal=0
    df.drop(columns=["MiscVal"], inplace=True)
    # 1453 houses have Pool=0
    df.drop(columns=["PoolArea"], inplace=True)
    # 1436 houses have 3SsnPorch=0
    df.drop(columns=["3SsnPorch"], inplace=True)
    # 1420 houses have LowQualFinSF=0
    df.drop(columns=["LowQualFinSF"], inplace=True)

    ### Ignore HalfBathrooms ####
    df.drop(columns=["BsmtHalfBath", "HalfBath"], inplace=True)

    #### Ignore time-related features ####
    # We mainly care about the PHYSICAL properties of the houses
    df.drop(columns=["YearRemodAdd", "YrSold", "YearBuilt"], inplace=True)
    df.drop(columns=["GarageYrBlt", "MoSold"], inplace=True)

    #### Multiple Features regarding the Basement are multi-colinear ####
    # Add the ratio of completion of the basement as a feature
    assert (df["TotalBsmtSF"] == df["BsmtUnfSF"] + df["BsmtFinSF1"] + df["BsmtFinSF2"]).all()
    has_basement = df["TotalBsmtSF"]>0
    df.insert(0, "BsmtPercFin", (has_basement).astype(int))
    df.loc[has_basement, "BsmtPercFin"] = 1 - df.loc[has_basement, "BsmtUnfSF"]/df.loc[has_basement, "TotalBsmtSF"]
    # Drop other features that involve the basement
    df.drop(columns=["TotalBsmtSF"], inplace=True)
    df.drop(columns=["BsmtUnfSF"], inplace=True)
    df.drop(columns=["BsmtFinSF1"], inplace=True)
    df.drop(columns=["BsmtFinSF2"], inplace=True)

    #### Almost Perfect multi-colinearity GrLivArea=1st + 2nd floors ####
    assert np.isclose(df["GrLivArea"], df["1stFlrSF"] + df["2ndFlrSF"]).mean() > 0.95
    df.drop(columns=["GrLivArea"], inplace=True)
    
    # Remove correlated/redundant features
    if remove_correlations:
        #### High Spearman Correlation ####
        # High correlation of 0.85 with GarageArea
        df.drop(labels=["GarageCars"], axis=1, inplace=True)

        # High correlation >0.6 with BsmtPercFin
        df.drop(columns=["BsmtFullBath"], inplace=True)

        # High correlation >0.6 with BedroomAbvGrd
        df.drop(columns=["TotRmsAbvGr"], inplace=True)

        # High correlation ~0.6 with OverallQual
        df.drop(columns=["FullBath"], inplace=True)

    # # Solve the weird issue with YearBuild and YearRemodAdd
    # bool_idx = df["YearRemodAdd"]==1950
    # df.loc[bool_idx, "YearRemodAdd"] = df.loc[bool_idx, "YearBuilt"]
    # df.drop(columns=["YearBuilt"], inplace=True)

    # Process the target (careful here)
    # df = df[df["SalePrice"] < 500000]
    # df = df[df["SalePrice"] > 50000]
    # df['SalePrice'] = np.log1p(df['SalePrice'])

    # Determine the ordering of the features
    if remove_correlations:
        feature_names = \
            ['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea',
            'BsmtPercFin', '1stFlrSF', '2ndFlrSF',
            'BedroomAbvGr',  'KitchenAbvGr', 'Fireplaces', 'GarageArea',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

        feature_types = [
            "sparse_num",
            "num_int",
            "num_int",
            "sparse_num",
            "percent",
            "sparse_num",
            "sparse_num",
            "num_int",
            "num_int",
            "num_int",
            "sparse_num",
            "sparse_num",
            "sparse_num",
            "sparse_num",
            "sparse_num",
        ]

    else:
        feature_names = \
            ['LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea',
            'BsmtPercFin', '1stFlrSF', '2ndFlrSF', 'FullBath', 
            'BsmtFullBath', 'BedroomAbvGr', 'TotRmsAbvGr', 'KitchenAbvGr', 
            'Fireplaces', 'GarageArea', 'GarageCars',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']

        feature_types = [
            "sparse_num",
            "num_int",
            "num_int",
            "sparse_num",
            "percent",
            "sparse_num",
            "sparse_num",
            "num_int",
            "num_int",
            "num_int",
            "num_int",
            "num_int",
            "num_int",
            "sparse_num",
            "num_int",
            "sparse_num",
            "sparse_num",
            "sparse_num",
            "sparse_num",
        ]
    
    if submission:
        df = df[feature_names]
        X = df.to_numpy()
        y = None
    else:
        df = df[feature_names+['SalePrice']]
        X = df.to_numpy()[:, :-1]
        y = np.log(df.to_numpy()[:, -1])
    
    features = Features(X, feature_names, feature_types)
    if submission:
        return X, y, features, Id
    else:
        return X, y, features



DATASET_MAPPING = {
    "bike": get_data_bike,
    "california": get_data_california_housing,
    "adult_income" : get_data_adults,
    "compas": get_data_compas,
    "marketing": get_data_marketing,
    "kin8nm": get_data_kin8nm,
    "default_credit": get_data_credit,
    "kaggle_houses" : get_data_kaggle_housing,
}


TASK_MAPPING = {
    "bike": "regression",
    "california" : "regression",
    "adult_income": "classification",
    "compas": "classification",
    "marketing": "classification",
    "kin8nm": "regression",
    "default_credit": "classification",
    "kaggle_houses": "regression"
}


if __name__ == "__main__":
    for name, import_fun in DATASET_MAPPING.items():
        print(f"Loading {name}\n")
        X, y, features = import_fun()
        print(X.shape)
        print(y.shape)
        features.summary()
        print("\n\n")
