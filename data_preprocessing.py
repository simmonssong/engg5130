import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler


runs = pd.read_csv("datasets/runs.csv")
# print(runs.head())

races = pd.read_csv("datasets/races.csv")

data = runs

weather = pd.read_csv("datasets/weather.csv")


def check_features():
    print(data.head())
    print(data.shape)

    # print(races["win_combination2"].unique())
    """
    Race features
    """
    # feature_name = ["place_combination1", "place_dividend1", "win_combination1"]

    # feature_name = ["place_combination1", "place_dividend1", "win_combination1", "win_dividend1"]
    # feature_name = ["race_class"]

    # feature_name = ["place_combination2", "place_dividend2", "win_combination2", "win_dividend2"]
    # feature_name = ["place_combination1", "place_dividend1", "win_combination1"]

    """
    Runs features
    """

    # feature_name = ["horse_country", "horse_type"]
    # feature_name = ["win_combination1"]
    # feature_name = ["place_dividend1"]
    feature_name = ["race_id"]
    # print(data.loc[:, feature_name].head())
    # print(data.loc[:, feature_name].value_counts())
    # print(data.loc[:, feature_name].isnull().sum())

    # print((races["place_combination2"] == races["win_combination2"]).value_counts())

    # print((races["place_dividend2"] >= races["win_dividend2"]).value_counts())

    print(data.groupby(feature_name)["horse_id"].count().value_counts())


# check_features()

# 2. Merge Data


def merge_data(features=["race_id", "horse_id", "won", "horse_age", "horse_country", "horse_type",
                         "horse_rating", "horse_gear", "declared_weight", "actual_weight", "draw", "win_odds", "place_odds", "venue", "config",
                         "surface", "distance", "going", "horse_ratings", "race_class", "mean_temp",
                         "max_Temp", "mean_amount_of_cloud", "mean_dew_point_temp", "mean_wet_bulb_temp", "min_temp", "pressure", "RH"]):
    race_weather = pd.merge(races, weather, left_on="date", right_on="date")

    race_weather_run = pd.merge(
        runs, race_weather, left_on="race_id", right_on="race_id")

    # print(races.shape)
    # print(weather.shape)
    # print(race_weather.shape)
    # print(runs.shape)
    # print(race_weather_run[features].dropna().shape)

    # print(race_weather_run[features].head())

    # print(race_weather_run[features].columns)

    # print(race_weather_run[features].dropna().isnull().sum().sort_values(ascending=False))

    # race_weather_run[features].dropna().to_csv("datasets/post_runs.csv")

    return race_weather_run[features].dropna()


# merge_data()

def check_distribution():
    run_race_weather = merge_data()
    subset_attributes=['horse_age', 'horse_rating', 'declared_weight',
                                          'actual_weight', 'draw', 'win_odds', 'place_odds',
                                          'surface', 'distance', 'race_class', 'mean_temp', 'RH']

    # subset_attributes = []
    # print(run_race_weather["draw"] == run_race_weather["horse_no"])

    # print(run_race_weather["draw"].value_counts())

    # print(run_race_weather["horse_no"].value_counts())

    Winners = round(run_race_weather[run_race_weather['won']==1][subset_attributes].describe(),2)
    Not_Winners = round(run_race_weather[run_race_weather['won']==0][subset_attributes].describe(),2)

    # result = pd.concat([Winners, Not_Winners], axis=1, keys=['Winners', 'Not Winners'])
    print(Winners)

    print(Not_Winners)
    
# check_distribution()

def check_feature_correlation():
    data = merge_data()
    print(data.shape)
    data = data[data["draw"] != 15]
    print(data.shape)
    # attributes = ['horse_age', 'horse_rating', 'declared_weight',
    #                                       'actual_weight', 'draw', 'win_odds', 'place_odds',
    #                                       'surface', 'distance', 'race_class', 'mean_temp', 'RH']
    # data = data[attributes]

    f, ax = plt.subplots(figsize=(15, 10))
    corr = data.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Correlation Heatmap', fontsize=14)

    plt.savefig("results/correlation.png")

# check_feature_correlation()

final_p_attributes = ["won", "horse_rating",  "declared_weight", "actual_weight", "draw", "win_odds", "place_odds"]

attributes_categorical = ['horse_country', 'horse_type',
                       'horse_gear','venue', 'config','going','surface','race_class']

attributes_numerical = ['horse_age','horse_rating', 'declared_weight',
                     'actual_weight','draw', 'win_odds', 'place_odds', 'distance', "mean_temp", "mean_amount_of_cloud", "pressure", "RH"]

final_all_attributes = ["won", "horse_age", "horse_country", "horse_type",
                         "horse_rating", "horse_gear", "declared_weight", "actual_weight", "draw", "win_odds", "place_odds", "venue", "config",
                         "surface", "distance", "going", "horse_ratings", "race_class", 
                         "mean_temp", "mean_amount_of_cloud", "pressure", "RH"]


def check_row(final_attributes):
    data = merge_data()
    data = data[data["draw"] != 15]
    data = data[final_attributes]


# check_row()


def encode_categorical():
    data = merge_data()
    data = data[data["draw"] != 15]
    data = data[final_all_attributes]
    data_c = data[attributes_categorical]
    data_c = pd.get_dummies(data_c.apply(lambda col: col.astype('category')))
    print(data_c.shape)
    # for c in attributes_categorical:
    #     data_c = data[[c]]
    #     data_c = pd.get_dummies(data_c.apply(lambda col: col.astype('category')))
    #     print(data_c.columns)
    #     print("******")
    return data_c

# encode_categorical()

def box_fig_numerical(data_a, data_o, attr):
    sns.set(style="ticks")
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # plt.title(attr + "Distribution", fontsize=14)
    # Creation of 1st axis
    sns.boxplot(data=data_o, ax=ax[0])
    ax[0].legend(loc='upper right')
    ax[0].set_title(attr + " Before Normalization", fontsize=14)
    
    # Creation of 2nd axis
    sns.boxplot(data=data_a,ax=ax[1])
    ax[1].set_title(attr + " After Normalization", fontsize=14)
    # ax[1].set(ylim=(0, 40))

    # Close the empty Figure 2 created by seaborn.
    plt.savefig("results/norm_"+ attr + ".png")
    plt.close(2)
    


def normalize_numerical():
    data = merge_data()
    data = data[data["draw"] != 15]
    data = data[final_all_attributes]
    data_o = data[attributes_numerical]
    data_c = data[attributes_numerical]

    ct = ColumnTransformer([], remainder=StandardScaler())
    tmp = ct.fit_transform(data_c)
    # keep feature names
    for (col_index, col_name) in enumerate(list(data_c.columns)):
        data_c[col_name] = tmp[:, col_index]
        print(col_name)
        box_fig_numerical(tmp[:, col_index], data_o.loc[:, col_name], col_name)
        
    print(data_c.shape)

    
    return data_c


normalize_numerical()

def check_odds_distribution(data):
    sns.set(style="ticks")
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.boxplot(x="won", y="win_odds", data=data, ax=ax[0])
    ax[0].legend(loc='upper right')
    ax[0].set_title(
        "Win odds distribution for non-winners and winners", fontsize=14)

    sns.boxplot(x="won", y="place_odds", data=data, ax=ax[1])
    ax[1].set_title(
        "Place odds boxplot for non-winners and winners", fontsize=14)
    ax[1].set(ylim=(0, 40))

    plt.savefig("results/win_place_odds_dist.png")
    plt.close(2)


def remove_inter_match_information(runs):
    runs_data = runs[['race_id', 'won', 'horse_age', 'horse_country', 'horse_type', 'horse_rating',
                      'horse_gear', 'declared_weight', 'actual_weight', 'draw', 'win_odds',
                      'place_odds', 'horse_id']]

    