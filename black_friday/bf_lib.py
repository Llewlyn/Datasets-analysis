import numpy as np
import pandas as pd


def encode_features(df):
    # Gender
    df['sex'] = pd.get_dummies(df['Gender']).M  # 0 M 1 F
    df.drop(['Gender'], axis=1, inplace=True)

    # Age
    ages = df['Age'].unique()
    ages2 = [age.split('-')[0] for age in ages]
    ages2 = [int(age.split('+')[0]) for age in ages2]
    sort_ixs = np.argsort(ages2)
    age_rule = {ages[ix]: sort_ix for ix, sort_ix in enumerate(sort_ixs)}
    df['age'] = df.Age.map(age_rule)
    df.drop(['Age'], axis=1, inplace=True)

    # Occupation
    occ_df = pd.get_dummies(df.Occupation, prefix='occ').iloc[:, :-1]
    df = pd.concat([df, occ_df], axis=1)
    df = df.drop('Occupation', axis=1)

    # City Category
    cc_df = pd.get_dummies(data=df.City_Category, prefix='citycat').iloc[:, :-1]
    df = pd.concat([df, cc_df], axis=1)
    df = df.drop('City_Category', axis=1)

    # Stays in current city years
    years_rule = {val: int(val.split('+')[0]) for val in df.Stay_In_Current_City_Years.unique()}
    df['years'] = df.Stay_In_Current_City_Years.map(years_rule)
    df = df.drop('Stay_In_Current_City_Years', axis=1)

    # Product categories
    pc1_df = pd.get_dummies(df.Product_Category_1, prefix='pc1')
    df = pd.concat([df, pc1_df], axis=1)
    df.drop('Product_Category_1', axis=1, inplace=True)
    pc2_df = pd.get_dummies(df.Product_Category_2, prefix='pc2', dummy_na=True)
    pc3_df = pd.get_dummies(df.Product_Category_3, prefix='pc3', dummy_na=True)
    df = pd.concat([df, pc2_df, pc3_df], axis=1)
    df.drop(['Product_Category_2', 'Product_Category_3'], axis=1, inplace=True)
    df.drop(['prod_cat_2', 'prod_cat_3'], axis=1, inplace=True)

    return df


def normalize_features(df):
    import sklearn.preprocessing as skpp

    # Standard scaling of purchase
    ss = skpp.StandardScaler()
    purchase = np.reshape(df.Purchase.values, (-1, 1))
    ys = ss.fit_transform(purchase)
    df.drop('Purchase', axis=1, inplace=True)

    # min max scaling num_items
    mm = skpp.MinMaxScaler()
    df['num_items_norm'] = mm.fit_transform(np.reshape(df.num_items.values, (-1, 1)))
    df.drop('num_items', axis=1, inplace=True)

    # min max scaling age (band)
    mm = skpp.MinMaxScaler()
    df['age_norm'] = mm.fit_transform(np.reshape(df.age.values, (-1, 1)))
    df.drop('age', axis=1, inplace=True)

    # min max scaling of years in city
    mm = skpp.MinMaxScaler()
    df['years_norm'] = mm.fit_transform(np.reshape(df.years.values, (-1, 1)))
    df.drop('years', axis=1, inplace=True)

    return df
