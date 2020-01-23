import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import math
import numpy as np
import plotly.graph_objs as go

from GowerDistance import gower_distances
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import chebyshev
import itertools

import xgboost

import shap
from shap_calculator import Explainer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from collections import defaultdict
from collections import Counter

import pickle

"""

Initializing functions

"""


def feature_importance_bar_exact(shap_value, lim):
    if shap_value >= 0:
        color = '#0DB5E6'
    else:
        color = '#ffa54c'
    # Trace definition
    hoverlabel = {
        'bordercolor': 'white',
        'font': {'size': 10},
    }
    trace = go.Bar(x=[shap_value],
                   y=[''],
                   orientation='h',
                   hoverinfo='x',
                   hoverlabel=hoverlabel,
                   marker={'color': color},
                   )
    # Layout definition
    xaxis = {
        'range': [-lim, lim],
        'fixedrange': True,
        'showgrid': False,
        'zeroline': False,
        'showline': False,
        'showticklabels': False,
        'hoverformat': '.2f'
    }
    yaxis = {
        'fixedrange': True,
        'showgrid': False,
        'zeroline': False,
        'showline': False,
        'showticklabels': False
    }
    margin = go.layout.Margin(l=0, r=0, t=0, b=0, pad=0)
    layout = go.Layout(yaxis=yaxis,
                       xaxis=xaxis,
                       margin=margin,
                       bargap=0,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')

    # Config definition
    config = {'displayModeBar': False,
              'showLink': False}

    return dcc.Graph(figure={'data': [trace],
                             'layout': layout},
                     config=config,
                     style={'height': '18px',
                            'width': '170px'})


def log_odd_to_prob(log_odd_list):
    prob_list = list()
    for element in log_odd_list:
        print(element)
        x = math.exp(element)/(1+math.exp(element))
        print(x)
        prob_list.append(x)

    return prob_list


def label_encode(df, cat_variables):
    dataframe = df.copy()
    label_dictionary = defaultdict(LabelEncoder)
    dataframe[cat_variables] = dataframe[cat_variables].apply(lambda x: label_dictionary[x.name].fit_transform(x))
    return dataframe, label_dictionary


def label_decode(df, cat_variables, label_dictionary):
    dataframe = df.copy()
    dataframe[cat_variables] = dataframe[cat_variables].apply(lambda x: label_dictionary[x.name].inverse_transform(x))
    return dataframe


def label_encode_new(df, cat_variables, label_dictionary):
    dataframe = df.copy()
    dataframe[cat_variables] = dataframe[cat_variables].apply(lambda x: label_dictionary[x.name].transform(x))
    return dataframe


# def get_alerts():
#     false_alerts_index = []
#
#     for i in range(0, 1000):
#         if y_pred_tf[i] == True and y_test[i] == False:
#             false_alerts_index.append(i)
#
#     false_alerts_display = X_test_display.iloc[false_alerts_index]
#     false_alerts = X_test.iloc[false_alerts_index]
#
#     return false_alerts_display, false_alerts


def get_shap(instance):
    shap_instance = np.empty((0, len(X.columns)))
    for i in range(0, len(instance.index)):
        values = explainer.standard(x=np.asarray(instance.iloc[i]), m=100, return_samples=True)
        shap_instance = np.append(shap_instance, [list(values.shap_values.values())], axis=0)
    return shap_instance


def get_abs_shap(instance):
    abs_shap_instance = np.empty((0, len(X.columns)))
    for i in range(0, len(instance.index)):
        values = explainer.standard(x=np.asarray(instance.iloc[i]), m=100, return_samples=True)
        abs_shap_instance = np.append(abs_shap_instance, [list(values.abs_shap_values.values())], axis=0)
    return abs_shap_instance


def get_prediction(instance):
    return model.predict_proba(instance)[:,1]


def get_perturbed_samples_univariate(alert_display, alert, features_to_perturb, num_of_samples):
    random_state = np.random

    stan_scale = StandardScaler()
    X_ss = stan_scale.fit(X_train)

    perturbed_samples = alert.copy()

    for i in range(0, num_of_samples - 1):
        perturbed_samples = perturbed_samples.append(alert)

    perturbed_samples.reset_index(drop=True, inplace=True)

    for column in features_to_perturb:

        if pd.api.types.infer_dtype(X_train_display[column]) == 'integer':
            normal_random = random_state.normal(0, 1, num_of_samples)
            random_num = abs(normal_random * stan_scale.scale_[X_train.columns.get_loc(column)] + alert[column].values)
            perturbed_samples[column] = [round(x) for x in random_num]
            perturbed_samples[column] = perturbed_samples[column].astype(int)

        elif pd.api.types.infer_dtype(X_train_display[column]) == 'floating':
            normal_random = random_state.normal(0, 1, num_of_samples)
            random_num = abs(normal_random * stan_scale.scale_[X_train.columns.get_loc(column)] + alert[column].values)
            perturbed_samples[column] = random_num

        elif pd.api.types.infer_dtype(X_train_display[column]) == 'categorical':
            counter = Counter(X_train[column])
            frequency = np.array(list(counter.values())) / len(X_train[column])
            random_cat = random_state.choice(list(counter.keys()), num_of_samples, p=frequency)
            perturbed_samples[column] = random_cat

        else:
            raise TypeError("The feature '{}' used for perturbation has datatype other than integer, float and "
                            "categorical".format(column))

    perturbed_samples_display = label_decode(perturbed_samples, cat_features, encode_dict)

    return perturbed_samples_display, perturbed_samples


def get_all_perturbed_samples_univariate(alert_display, alert, features_to_perturb):
    perturbed_samples_display = alert_display.copy()
    perturbed_samples = alert.copy()

    if len(features_to_perturb) == 1:
        for subset in itertools.combinations(features_to_perturb, 1):
            perturbed_subset_display, perturbed_subset = get_perturbed_samples_univariate(
                alert_display, alert, subset, 10)
            perturbed_samples_display = perturbed_samples_display.append(perturbed_subset_display)
            perturbed_samples = perturbed_samples.append(perturbed_subset)

    else:
        for i in range(1, len(features_to_perturb)+1):
            for subset in itertools.combinations(features_to_perturb, i):
                perturbed_subset_display, perturbed_subset = get_perturbed_samples_univariate(
                    alert_display, alert, subset, 10)
                perturbed_samples_display = perturbed_samples_display.append(perturbed_subset_display)
                perturbed_samples = perturbed_samples.append(perturbed_subset)

    perturbed_samples_display = perturbed_samples_display.iloc[1:]
    perturbed_samples = perturbed_samples.iloc[1:]

    perturbed_samples_display.drop_duplicates(inplace=True)
    perturbed_samples.drop_duplicates(inplace=True)

    perturbed_samples_display.reset_index(drop=True, inplace=True)
    perturbed_samples.reset_index(drop=True, inplace=True)

    return perturbed_samples_display, perturbed_samples


def get_perturbed_samples_uniform(alert_display, alert, features_to_perturb, num_of_samples):
    random_state = np.random

    stan_scale = StandardScaler()
    X_ss = stan_scale.fit(X_train)

    perturbed_samples = alert.copy()

    for i in range(0, num_of_samples - 1):
        perturbed_samples = perturbed_samples.append(alert)

    perturbed_samples.reset_index(drop=True, inplace=True)

    for column in features_to_perturb:

        if pd.api.types.infer_dtype(X_train_display[column]) == 'integer':
            random_num = random_state.uniform(X_train[column].max(), X_train[column].min(), num_of_samples)
            perturbed_samples[column] = [round(x) for x in random_num]
            perturbed_samples[column] = perturbed_samples[column].astype(int)

        elif pd.api.types.infer_dtype(X_train_display[column]) == 'floating':
            random_num = random_state.uniform(X_train[column].max(), X_train[column].min(), num_of_samples)
            perturbed_samples[column] = random_num

        elif pd.api.types.infer_dtype(X_train_display[column]) == 'categorical':
            counter = Counter(X_train[column])
            random_cat = random_state.choice(list(counter.keys()), num_of_samples)
            perturbed_samples[column] = random_cat

        else:
            raise TypeError(
                "The feature '{}' used for perturbation has datatype other than integer, float and categorical".format(
                    column))

    perturbed_samples_display = label_decode(perturbed_samples, cat_features, encode_dict)

    return perturbed_samples_display, perturbed_samples


def get_all_perturbed_samples_uniform(alert_display, alert, features_to_perturb):
    perturbed_samples_display = alert_display.copy()
    perturbed_samples = alert.copy()

    if len(features_to_perturb) == 1:
        for subset in itertools.combinations(features_to_perturb, 1):
            perturbed_subset_display, perturbed_subset = get_perturbed_samples_uniform(
                alert_display, alert, subset, 10)
            perturbed_samples_display = perturbed_samples_display.append(perturbed_subset_display)
            perturbed_samples = perturbed_samples.append(perturbed_subset)

    else:
        for i in range(1, len(features_to_perturb)+1):
            for subset in itertools.combinations(features_to_perturb, i):
                perturbed_subset_display, perturbed_subset = get_perturbed_samples_uniform(
                    alert_display, alert, subset, 10)
                perturbed_samples_display = perturbed_samples_display.append(perturbed_subset_display)
                perturbed_samples = perturbed_samples.append(perturbed_subset)

    perturbed_samples_display = perturbed_samples_display.iloc[1:]
    perturbed_samples = perturbed_samples.iloc[1:]

    perturbed_samples_display.drop_duplicates(inplace=True)
    perturbed_samples.drop_duplicates(inplace=True)

    perturbed_samples_display.reset_index(drop=True, inplace=True)
    perturbed_samples.reset_index(drop=True, inplace=True)

    return perturbed_samples_display, perturbed_samples


def generate_counterfactuals(perturbed_samples, perturbed_samples_pred, prob_threshold):
    counterfactauls = perturbed_samples.copy()
    #counterfactauls['probability_true'] = perturbed_samples_pred
    #counterfactauls = counterfactauls.loc[(counterfactauls['probability_true'] < prob_threshold[0])]
    counterfactauls.reset_index(drop=True, inplace=True)
    #counterfactauls.drop(['probability_true'], axis=1, inplace=True)

    counterfactuals_display = label_decode(counterfactauls, cat_features, encode_dict)

    return counterfactuals_display, counterfactauls


def calculate_gower_distance(alert_display, counterfactuals_display):
    all_in_one = pd.concat([X_train_display, counterfactuals_display], axis=0)
    D = gower_distances(all_in_one, alert_display)

    dist = []

    temp = D[len(X_train_display.index):].tolist()

    for i in temp:
        dist.append(i[0])

    return dist


def calculate_shap_distance(alert, counterfactuals):
    alert_counterfactuals = pd.concat([alert, counterfactuals], axis=0)
    alert_counterfactuals_shap = get_shap(alert_counterfactuals)

    dist = []

    for i in range(1, len(alert_counterfactuals.index)):
        dist.append(cityblock(alert_counterfactuals_shap[0], alert_counterfactuals_shap[i]))
        # sum = 0
        # for j in range(0, len(alert_counterfactuals_shap[i])):
        #     sum = sum + alert_counterfactuals_shap[0][j] - alert_counterfactuals_shap[i][j]
        #
        # dist.append(sum)

    return dist


def calculate_distance(alert_display, counterfactuals_display, cat_features):
    all_in_one = pd.concat([alert_display, counterfactuals_display], axis=0)
    all_in_one = pd.get_dummies(all_in_one, prefix=cat_features, columns=cat_features)

    all_in_one_ss = StandardScaler().fit_transform(all_in_one)

    dist = []

    for i in range(1, len(all_in_one.index)):
        dist.append(cityblock(all_in_one_ss[0], all_in_one_ss[i]))

    return dist


def calculate_numeric_shap_distance(alert, counterfactuals, cat_features):
    alert_counterfactuals = pd.concat([alert, counterfactuals], axis=0)
    all_in_one = pd.concat([alert_counterfactuals, X_train], axis=0)
    alert_shap = get_shap(alert)
    counterfactuals_shap = get_shap(counterfactuals)
    alert_counterfactuals_shap = np.vstack((alert_shap, counterfactuals_shap))
    all_in_one_shap = np.vstack((alert_counterfactuals_shap, shap_train))

    for cat in cat_features:
        all_in_one[cat] = all_in_one_shap[:, all_in_one.columns.get_loc(cat)]

    all_in_one_ss = StandardScaler().fit_transform(all_in_one)

    dist = []

    for i in range(1, len(alert_counterfactuals.index)):
        dist.append(cityblock(all_in_one_ss[0], all_in_one_ss[i]))

    return dist


def get_pareto_frontiers(probabilities, distance_values, alert_prob):
    pareto_check = np.vstack((np.array([probabilities]), np.array(distance_values))).T
    Xs = pareto_check[:, 0]
    Ys = pareto_check[:, 1]

    maxY = False
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front_left = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front_left[-1][1]:
                pareto_front_left.append(pair)
        else:
            if pair[1] <= pareto_front_left[-1][1]:
                pareto_front_left.append(pair)

    maxY = True
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front_right = [sorted_list[0]]
    # print(sorted_list)
    # print(pareto_front_right[-1][1])
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] <= pareto_front_right[-1][1]:
                pareto_front_right.append(pair)
        else:
            if pair[1] >= pareto_front_right[-1][1]:
                pareto_front_right.append(pair)

    pareto_front_right.reverse()

    # print(pareto_front_left)
    # print(pareto_front_right)

    pareto_front = pareto_front_left + pareto_front_right

    pareto_check = [list(l) for l in pareto_check]
    pareto_front = [list(l) for l in pareto_front]
    pareto_front_left = [list(l) for l in pareto_front_left]
    pareto_front_right = [list(l) for l in pareto_front_right]

    not_pareto_front = [list(l) for l in pareto_check]
    #not_pareto_front = [i for i in pareto_check + pareto_front if i not in pareto_check or i not in pareto_front]

    return pareto_front_left, pareto_front_right, not_pareto_front


def get_pareto_index(probabilities, distance_values, pareto_front):
    pareto_check = np.vstack((np.array([probabilities]), np.array(distance_values))).T
    pareto_check = pareto_check.tolist()
    pareto_index = []
    for i in range(0, pareto_front.shape[0]):
        pareto_index.append(pareto_check.index(list(pareto_front[i])))

    return pareto_index


def get_top_shap_contributors(instance, number_of_features):
    instance_shap = abs(get_shap(instance))
    feature_index = sorted(range(len(instance_shap[0])),
                           key=lambda i: instance_shap[0][i], reverse=True)[:(number_of_features)]
    return list(X.columns[feature_index])


# def tsne_plot(alert, counterfactuals):
#     alert_counterfactuals = pd.concat([alert, counterfactuals], axis=0)
#     tsne_plot = TSNE(n_components=2).fit_transform(alert_counterfactuals)
#     return tsne_plot
#
#
# def mds_plot(alert, counterfactuals):
#     alert_counterfactuals = pd.concat([alert, counterfactuals], axis=0)
#     mds_plot = MDS(n_components=2).fit_transform(alert_counterfactuals)
#     return mds_plot

#
# def iso_shap_plot(alert, counterfactuals):
#     alert_shap = get_shap(alert)
#     counterfactuals_shap = get_shap(counterfactuals)
#     alert_counterfactuals_shap = np.vstack((alert_shap, counterfactuals_shap))
#     tsne_shap_graph = iso_shap_plot_object.transform(alert_counterfactuals_shap)
#     return tsne_shap_graph

def iso_shap_plot(points):
    isoshap_graph = iso_shap_plot_object.transform(points)
    return isoshap_graph


# def mds_shap_plot(alert, counterfactuals):
#     alert_shap = get_shap(alert)
#     counterfactuals_shap = get_shap(counterfactuals)
#     alert_counterfactuals_shap = np.vstack((alert_shap, counterfactuals_shap))
#     tsne_shap_plot = MDS(n_components=2).fit_transform(alert_counterfactuals_shap)
#     return tsne_shap_plot
#
#
# def tsne_shap_plot(alert, counterfactuals):
#     alert_shap = get_shap(alert)
#     counterfactuals_shap = get_shap(counterfactuals)
#     print(len(counterfactuals_shap))
#
#
#     all_in_one = np.vstack((shap_train, alert_shap, counterfactuals_shap))
#     print(len(all_in_one[len(shap_train):, :]))
#     print(all_in_one[len(shap_train):, :])
#
#     tsne_shap_plot = PCA(n_components=2).fit_transform(all_in_one)
#
#     print(tsne_shap_plot[len(shap_train)-1:, :])
#     return tsne_shap_plot[len(shap_train)-1:, :]



"""

Initializing global stuffs

"""

pareto_prev = 2
neighbors_prev = 2
pareto_click_count = 0
neighbors_click_count = 0
plot_type = "tsne_shap"
distance_type = "shap"

age_range = np.arange(17, 91)
wc_range = np.arange(0, 7)
edu_num_range = np.arange(1, 17)
ms_range = np.arange(0, 7)
occ_range = np.arange(0, 14)
relationship_range = np.arange(0, 6)
race_range = np.arange(0, 5)
sex_range = np.arange(0, 2)
hpw_range = np.arange(1, 99)
country_range = np.arange(0, 40)

X, y = shap.datasets.adult(display=True)

X['TF'] = y
X = X[(X[X.columns] != ' ?').all(axis=1)]
X.reset_index(drop=True, inplace=True)
X.drop_duplicates(inplace=True)
X, y = X.loc[:, X.columns != 'TF'], X['TF']
y = np.array(y)

num_features = ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']
cat_features = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']

# convert float to int
X['Education-Num'] = X['Education-Num'].astype(int)
X['Age'] = X['Age'].astype(int)
X['Hours per week'] = X['Hours per week'].astype(int)

X_enc, encode_dict = label_encode(X, cat_features)
y_enc = y

X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, train_size=6000, test_size=2000, random_state=7)
X_train_a, X_train_b, y_train_a, y_train_b = train_test_split(X_train, y_train, train_size=4000, test_size=2000, random_state=7)
X_train_display, X_test_display, y_train_display, y_test_display = train_test_split(X, y, train_size=6000,
                                                                                    test_size=2000, random_state=7)

# model = LogisticRegression()
# model.fit(X_train_a, y_train_a)
# pickle.dump(model, open("Data/model_log_reg.p", "wb"))

model = pickle.load(open("Data/model_log_reg.p", "rb"))

y_pred_train = model.predict_proba(X_train)[:,1]
y_pred_tf_train = (y_pred_train >= 0.5)

confusion_matrix = []

for i in range(0, len(y_train)):
    if y_train[i] == True and y_pred_tf_train[i] == True:
        confusion_matrix.append("TP")
    elif y_train[i] == False and y_pred_tf_train[i] == False:
        confusion_matrix.append("TN")
    elif y_train[i] == False and y_pred_tf_train[i] == True:
        confusion_matrix.append("FP")
    else:
        confusion_matrix.append("FN")

explainer = Explainer(X = X_train, f = model)

# shap_train = np.empty((0, len(X.columns)))
# for i in range(0, len(X_train.index)):
#     values = explainer.standard(x=np.asarray(X_train.iloc[i]), m=100, return_samples=True)
#     shap_train = np.append(shap_train, [list(values.shap_values.values())], axis=0)
#
# shap_test = np.empty((0, len(X.columns)))
# for i in range(0, len(X_test.index)):
#     values = explainer.standard(x=np.asarray(X_test.iloc[i]), m=100, return_samples=True)
#     shap_test = np.append(shap_test, [list(values.shap_values.values())], axis=0)

# pickle.dump(shap_train, open("Data/shap_train_log_reg.p", "wb"))
# pickle.dump(shap_test, open("Data/shap_test_log_reg.p", "wb"))

shap_train = pickle.load(open("Data/shap_train_log_reg.p", "rb"))
shap_test = pickle.load(open("Data/shap_test_log_reg.p", "rb"))

# iso_shap_plot_object = Isomap(n_components=2).fit(shap_train)
# pickle.dump(iso_shap_plot_object, open("Data/isomap_log_reg.p", "wb"))

iso_shap_plot_object = pickle.load(open("Data/isomap_log_reg.p", "rb"))

# neighbors = NearestNeighbors(n_neighbors=20, metric='euclidean')
# neighbors.fit(shap_train)
# pickle.dump(neighbors, open("Data/neighbor_log_reg.p", "wb"))

neighbors = pickle.load(open("Data/neighbor_log_reg.p", "rb"))


all_alerts = [6419, 11635, 7562, 19962, 19038]
alert_display = X_test_display.loc[[all_alerts[0]]]
alert = X_test.loc[[all_alerts[0]]]
alert_y = y[all_alerts[0]]

perturbed_samples_display, perturbed_samples = \
    get_all_perturbed_samples_univariate(alert_display, alert, get_top_shap_contributors(alert, 2))

perturbed_samples_pred = get_prediction(perturbed_samples)
alert_pred = get_prediction(alert)

counterfactuals_display, counterfactuals = generate_counterfactuals(
    perturbed_samples, perturbed_samples_pred, alert_pred)

counterfactuals_pred = get_prediction(counterfactuals)

if distance_type == "gower":
    distance_values = calculate_gower_distance(alert_display, counterfactuals_display)
elif distance_type == "shap":
    distance_values = calculate_shap_distance(alert, counterfactuals)
elif distance_type == "numeric_shap":
    distance_values = calculate_numeric_shap_distance(alert, counterfactuals, cat_features)
else:
    distance_values = calculate_distance(alert_display, counterfactuals_display, cat_features)

pareto_front_left, pareto_front_right, not_pareto_front = get_pareto_frontiers(counterfactuals_pred, distance_values, alert_pred)

pareto_front_left_index = get_pareto_index(counterfactuals_pred, distance_values, np.array(pareto_front_left))
pareto_front_right_index = get_pareto_index(counterfactuals_pred, distance_values, np.array(pareto_front_right))

not_pareto_front_index = get_pareto_index(counterfactuals_pred, distance_values,
                                          np.array(not_pareto_front))

cf = counterfactuals.iloc[[0]]
cf_display = counterfactuals_display.iloc[[0]]


"""
STYLING
"""

colors = {
    'background': '#f6f6f6',
    'text-gray': '#727272'
}

# DIV STYLES
columnStyle = {'marginLeft': 5,
               'marginRight': 5,
               'backgroundColor': colors['background'],
               'paddingLeft': 20,
               'paddingRight': 20,
               'paddingBottom': 20,
               'height': '93vh',
               'overflow': 'auto'}

AlertColumnStyle = {'marginLeft': 0,
                   'paddingLeft': 20,
                   'paddingRight': 20,
                   'paddingBottom': 20,
                   'width': '15%',
                   'backgroundColor': 'rgb(56, 56, 56)',
                   'min-height': '100%',
                   'float': 'left'}

LeftColumnStyle = {'marginLeft': 0,
                   'paddingLeft': 20,
                   'paddingRight': 20,
                   'paddingBottom': 20,
                   'width': '45%',
                   'backgroundColor': 'rgb(245, 245, 245)',
                   'min-height': '100%',
                   'float': 'left'}

RightColumnStyle = {'marginLeft': 20,
                    'paddingLeft': 20,
                    'paddingRight': 20,
                    'paddingBottom': 20,
                    'width': '30%',
                    'backgroundColor': 'rgb(255, 255, 255)',
                    'float': 'right'}

# Table CSS

table_text_align = {'text-align': 'center',
                    'vertical-align': 'middle'}

table_text_align_bold = {'text-align': 'center',
                         'vertical-align': 'middle',
                         'font-weight': 'bold'}

table_row_phase_out = {'font-size': '1.5rem',
                       'marginTop': '10px',
                       'opacity': 0.4}

table_row_feature_change = {'font-size': '1.5rem',
                            'marginTop': '10px',
                            'background-color': '#c7e9b4'}

table_row_shap_change = {'font-size': '1.5rem',
                         'marginTop': '10px',
                         'background-color': '#ffffcc'}

# Confusion matrix colors
opacity = 0.5
cat_colors = {'TP': 'rgba(159, 211, 86, %s)' % opacity,
              'TN': 'rgba(13, 181, 230, %s)' % opacity,
              'FP': 'rgba(177, 15, 46, %s)' % opacity,
              'FN': 'rgba(255, 165, 76, %s)' % opacity}


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(children=[
        html.H4("Alerts", style={'color': 'White'}),
        html.Br(),
        html.Div([
            dcc.RadioItems(id='selected-alert',
                           options=[{'value': '1', 'label': 'Alert 1'},
                                    {'value': '2', 'label': 'Alert 2'},
                                    {'value': '3', 'label': 'Alert 3'},
                                    {'value': '4', 'label': 'Alert 4'},
                                    {'value': '5', 'label': 'Alert 5'},
                                    # {'value': '6', 'label': 'Alert 6'},
                                    # {'value': '7', 'label': 'Alert 7'},
                                    # {'value': '8', 'label': 'Alert 8'},
                                    # {'value': '9', 'label': 'Alert 9'},
                                    # {'value': '10', 'label': 'Alert 10'}
                                    ],
                           value='1', labelStyle={'paddingLeft': 7, 'color': 'White'},
                           style={'paddingTop': 9})
        ])
    ], style=AlertColumnStyle),
    html.Div(children=[
        html.H1("Alert Post-processing using Counterfactuals"),
        html.Br(),
        html.Br(),
        # html.Div([
        #     dcc.Markdown(['Enter number of features to perturb  :  '], style={'float': 'left', 'paddingTop': 7}),
        #     dcc.Input(id='perturb-input', value=2, style={'float': 'left', 'top': 20, 'width': 60}),
        #     html.Br(style={'clear': 'left'}),
        #     dcc.Markdown(['Select perturbation technique  :  '], style={'float': 'left', 'paddingTop': 7}),
        #     dcc.RadioItems(id='perturb-technique', options=[{'label': 'Univariate', 'value': 'Univariate'},
        #                                                     {'label': 'Uniform', 'value': 'Uniform'}],
        #                    value='Univariate', labelStyle={'display': 'inline-block', 'paddingLeft': 7},
        #                    style={'float': 'left', 'paddingTop': 9}),
        #     # html.Br(style={'clear': 'left'}),
        #     html.Button(id='perturb-button', n_clicks=0, children='Perturb',
        #                 style={'margin-left': 40, 'margin-bottom': 20})
        # ]),
        html.Br(style={'clear': 'left'}),
        # html.Br(),
        html.Div(id='alert-score'),
        html.Div(id='counter-score'),
        html.Center([
            html.Table(id='alert-cf-table')
        ])
    ], style=LeftColumnStyle),

    html.Div(children=[
        dcc.Graph(id='pareto'),
        dcc.Graph(id='neighbors')
    ], style=RightColumnStyle),

    # html.Div(id='number-of-perturb', style={'display': 'none'}, children=2),
    html.Div(id='selected-cf', style={'display': 'none'}, children=0)

], style={'display': 'flex'})

# @app.callback([Output('number-of-perturb', 'children')],
#               [Input('perturb-button', 'n_clicks')],
#               [State('perturb-input', 'value'),
#                State('perturb-technique', 'value')])
# def update_counterfactual(n_clicks, input_value, perturb_technique):
#     global alert
#     global alert_display
#     global alert_pred
#     global counterfactuals
#     global counterfactuals_display
#     global perturbed_samples
#     global perturbed_samples_display
#     global pareto_front_left
#     global pareto_front_right
#     global not_pareto_front
#     global pareto_front_left_index
#     global pareto_front_right_index
#     global not_pareto_front_index
#     global distance_values
#     global counterfactuals_pred
#
#     if n_clicks is not None:
#         if perturb_technique == 'Univariate':
#             perturbed_samples_display, perturbed_samples = \
#                 get_all_perturbed_samples_univariate(alert_display, alert, get_top_shap_contributors(alert,
#                                                                                                      int(input_value)))
#         else:
#             perturbed_samples_display, perturbed_samples = \
#                 get_all_perturbed_samples_uniform(alert_display, alert, get_top_shap_contributors(alert,
#                                                                                                      int(input_value)))
#
#         perturbed_samples_pred = get_prediction(perturbed_samples)
#         alert_pred = get_prediction(alert)
#
#         counterfactuals_display, counterfactuals = generate_counterfactuals(
#             perturbed_samples, perturbed_samples_pred, alert_pred)
#
#         counterfactuals_pred = get_prediction(counterfactuals)
#
#         #plot = iso_shap_plot(alert, counterfactuals)
#
#         if distance_type == "gower":
#             distance_values = calculate_gower_distance(alert_display, counterfactuals_display)
#         elif distance_type == "shap":
#             distance_values = calculate_shap_distance(alert, counterfactuals)
#         elif distance_type == "numeric_shap":
#             distance_values = calculate_numeric_shap_distance(alert, counterfactuals, cat_features)
#         else:
#             distance_values = calculate_distance(alert_display, counterfactuals_display, cat_features)
#
#         pareto_front_left, pareto_front_right, not_pareto_front = get_pareto_frontiers(counterfactuals_pred,
#                                                                                        distance_values, alert_pred)
#
#         pareto_front_left_index = get_pareto_index(counterfactuals_pred, distance_values, np.array(pareto_front_left))
#         pareto_front_right_index = get_pareto_index(counterfactuals_pred, distance_values, np.array(pareto_front_right))
#
#         not_pareto_front_index = get_pareto_index(counterfactuals_pred, distance_values,
#                                                   np.array(not_pareto_front))
#
#         return [input_value]


@app.callback([Output('pareto', 'figure')],
              [Input('selected-alert', 'value')])
def update_pareto_plot(selected_alert):

    global alert
    global alert_display
    global alert_pred
    global counterfactuals
    global counterfactuals_display
    global perturbed_samples
    global perturbed_samples_display
    global pareto_front_left
    global pareto_front_right
    global not_pareto_front
    global pareto_front_left_index
    global pareto_front_right_index
    global not_pareto_front_index
    global counterfactuals_pred
    global neighbors

    alert_index = selected_alert
    alert = X_test.loc[[all_alerts[int(alert_index) - 1]]]
    alert_display = X_test_display.loc[[all_alerts[int(alert_index) - 1]]]

    def calculate_cf(alert, attribute, attribute_range):
        df_cf = alert.copy()
        cf_prob = []
        for i in attribute_range:
            cf = alert.copy()
            cf[attribute] = i
            cf_prob.append(model.predict_proba(cf.iloc[[0]])[0][1])
            df_cf = df_cf.append(cf)
        df_cf.reset_index(inplace=True, drop=True)
        df_cf.drop([0], inplace=True)
        df_cf.reset_index(inplace=True, drop=True)
        df_cf_display = label_decode(df_cf, cat_features, encode_dict)
        return df_cf, df_cf_display

    alert_abs_shap = get_abs_shap(alert)[0]
    max_abs_shap_index = np.argmax(alert_abs_shap)

    print(alert_index)

    if int(alert_index) in [3, 10]:
        plot_range = age_range
        plot_feature = 'Age'
        col_index = 0
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Age', age_range)
    elif int(alert_index) in [4]:
        plot_range = wc_range
        plot_feature = 'Workclass'
        col_index = 1
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Workclass', wc_range)
    elif int(alert_index) in [1,5]:
        plot_range = edu_num_range
        plot_feature = 'Education-Num'
        col_index = 2
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Education-Num', edu_num_range)
    elif int(alert_index) in [8]:
        plot_range = ms_range
        plot_feature = 'Marital Status'
        col_index = 3
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Marital Status', ms_range)
    elif int(alert_index) in [9]:
        plot_range = occ_range
        plot_feature = 'Occupation'
        col_index = 4
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Occupation', occ_range)
    elif int(alert_index) in [6]:
        plot_range = relationship_range
        plot_feature = 'Relationship'
        col_index = 5
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Relationship', relationship_range)
    elif int(alert_index) in []:
        plot_range = race_range
        plot_feature = 'Race'
        col_index = 6
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Race', race_range)
    elif int(alert_index) in []:
        plot_range = sex_range
        plot_feature = 'Sex'
        col_index = 7
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Sex', sex_range)
    elif int(alert_index) in [2, 7]:
        plot_range = hpw_range
        plot_feature = 'Hours per week'
        col_index = 10
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Hours per week', hpw_range)
    elif int(alert_index) in []:
        plot_range = country_range
        plot_feature = 'Country'
        col_index = 11
        counterfactuals, counterfactuals_display = calculate_cf(alert, 'Country', country_range)

    counterfactuals_shap = get_shap(counterfactuals)
    shap_value_plot = counterfactuals_shap[:, col_index]

    counterfactuals_neighbors_distance, counterfactuals_neighbors_index = neighbors.kneighbors(counterfactuals_shap,
                                                           n_neighbors=15, return_distance=True)

    TS = []

    for j in range(0, len(counterfactuals_neighbors_distance)):
        total_dist = 0
        dist = 0
        for i in range(0, len(counterfactuals_neighbors_index[j])):
            test_index = counterfactuals_neighbors_index[j][i]
            test_distance = counterfactuals_neighbors_distance[j][i]
            if counterfactuals_neighbors_distance[j][i] == 0.0:
                test_distance = np.min(counterfactuals_neighbors_distance[j][0:][
                                           np.nonzero(counterfactuals_neighbors_distance[j][0:])])
            if confusion_matrix[test_index] == 'TP':
                # print("TP")
                total_dist = total_dist + (1 / test_distance)
                dist = dist + (1 / test_distance)
            elif confusion_matrix[test_index] == 'TN':
                # print("TN")
                total_dist = total_dist + (1 / test_distance)
                dist = dist + (1 / test_distance)
            elif confusion_matrix[test_index] == 'FP':
                # print("FP")
                total_dist = total_dist + (1 / test_distance)
            else:
                # print("FN")
                total_dist = total_dist + (1 / test_distance)
        score = dist / total_dist
        TS.append(score)

    print(TS)

    return [{
        'data': [
            go.Bar(
                x=counterfactuals_display[plot_feature],
                y=shap_value_plot.tolist(),
                customdata=np.arange(len(plot_range)).tolist(),
                text=TS,
                marker={
                    'color': TS,
                    'colorscale': 'Viridis',
                    'line': {'width': 0.5, 'color': 'white'},
                    'colorbar': {'title': 'Trust Score', 'x': 1.3}
                },
            )
        ],
        'layout':
            go.Layout(
                title='Feature SHAP v/s Feature Value',
                xaxis={'title': plot_feature},
                yaxis={'title': 'SHAP Value'},
                hovermode='closest',
                clickmode='event+select',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
    }]


@app.callback([Output('neighbors', 'figure')],
              [Input('pareto', 'clickData')])
def update_neighbor_plot(pareto_clickdata):
    global alert
    global alert_display
    global counterfactuals
    global counterfactuals_display
    global perturbed_samples
    global perturbed_samples_display
    global pareto_front
    global not_pareto_front
    global pareto_front_index
    global not_pareto_front_index
    global distance_values
    global neighbors_plot
    global counterfactuals_pred

    cf_index = pareto_clickdata['points'][0]['customdata']
    cf = counterfactuals.iloc[[cf_index]]
    cf_display = counterfactuals_display.iloc[[cf_index]]
    cf_shap = get_shap(cf)[0]

    counterfactuals_neighbors_index = neighbors.kneighbors([cf_shap],
                                                           n_neighbors=15, return_distance=False)

    neighbors_shap = np.array(shap_train)[counterfactuals_neighbors_index[0], :]

    neighbors_plot = iso_shap_plot(neighbors_shap)
    selected_cf_plot = iso_shap_plot([cf_shap])

    TP = []
    TN = []
    FP = []
    FN = []

    for i in range(0, neighbors_plot.shape[0]):
        test_index = counterfactuals_neighbors_index[0][i]
        if confusion_matrix[test_index] == 'TP':
            TP.append(i)
        elif confusion_matrix[test_index] == 'TN':
            TN.append(i)
        elif confusion_matrix[test_index] == 'FP':
            FP.append(i)
        else:
            FN.append(i)

    traces = []

    for row_index, perf in zip([TP, TN, FP, FN], ['TP', 'TN', 'FP', 'FN']):
        scatter = go.Scatter(
                x=list(np.array(neighbors_plot)[row_index, [0]]),
                y=list(np.array(neighbors_plot)[row_index, [1]]),
                mode='markers',
                marker={'line': {'width': 0.5, 'color': cat_colors[perf]},
                        'color': cat_colors[perf],
                        'colorscale': [[0, 'rgba(75, 75, 75, 1)'], [1, 'rgba(75, 75, 75, 1)']],
                        'cmin': 0,
                        'cmax': 1,
                        'size': 10},
                showlegend=True,
                name=perf
        )
        traces.append(scatter)

    counter = go.Scatter(
        x=[selected_cf_plot[0][0]],
        y=[selected_cf_plot[0][1]],
        mode='markers',
        opacity=1,
        marker={
            'symbol': 'triangle-up',
            'size': 15,
            'color': '#000000',
            'line': {'width': 0.5, 'color': 'white'}
        },
        name='Selected CF'
    )
    traces.append(counter)

    return [
        {
            'data': traces,
            'layout': go.Layout(
                title='Nearest Neighbor Plot',
                legend={'x': 1,
                        'y': 1.2},
                xaxis={'fixedrange': False,
                       'showgrid': True,
                       'zeroline': False,
                       'showline': False,
                       'showticklabels': False,
                       },
                yaxis={'fixedrange': False,
                       'showgrid': True,
                       'zeroline': False,
                       'showline': False,
                       'showticklabels': False,
                       },
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        }
    ]


@app.callback([Output('selected-cf', 'children')],
              [Input('pareto', 'clickData')])
def update_selected_cf(pareto_clickdata):

    try:
        selected_point = pareto_clickdata['points'][0]['customdata']
    except TypeError:
        print("No points selected yet in Pareto plot")
        selected_point = 0

    return [selected_point]


@app.callback([Output('alert-cf-table', 'children'),
               Output('alert-score', 'children'),
               Output('counter-score', 'children')],
              [Input('selected-alert', 'value'),
               Input('selected-cf', 'children')])
def click_and_table(selected_alert, selected_cf):
    global alert
    global alert_display
    global counterfactuals
    global counterfactuals_display
    global cf
    global cf_display

    cf_index = selected_cf
    alert_index = selected_alert
    alert = X_test.loc[[all_alerts[int(alert_index) - 1]]]
    alert_display = X_test_display.loc[[all_alerts[int(alert_index) - 1]]]
    alert_shap = get_shap(alert)[0]
    cf = counterfactuals.iloc[[cf_index]]
    cf_display = counterfactuals_display.iloc[[cf_index]]
    cf_shap = get_shap(cf)[0]
    table = list()
    table.append(html.Tr([html.Th(col, style=table_text_align) for col in ['Alert Contribution', 'Alert Value',
                                                                           'Features', 'Counterfactual Value',
                                                                           'Counterfactual Contribution']]))

    changed = []
    unchanged = []
    shap_changed = []
    shap_unchanged = []

    for i in range(0, len(alert.columns)):

        if cf.iloc[0][i] == alert.iloc[0][i]:
            unchanged.append(i)
        else:
            changed.append(i)

        if (abs(cf_shap[i] - alert_shap[i])) <= 0.05:
            shap_unchanged.append(i)
        else:
            shap_changed.append(i)

    all_changed = changed + list(set(shap_changed) - set(changed))
    final_order = all_changed + list(set(range(0, len(alert.columns))) - set(all_changed))

    for i in final_order:
        if i in changed:
            row_style = table_row_feature_change
        elif i in shap_changed:
            row_style = table_row_shap_change
        else:
            row_style = table_row_phase_out
        table.append(html.Tr([
            html.Td(feature_importance_bar_exact(alert_shap[i], 0.5)),
            html.Td(alert_display.iloc[0][i], style=table_text_align),
            html.Td(X.columns[i], style=table_text_align_bold),
            html.Td(cf_display.iloc[0][i], style=table_text_align),
            html.Td(feature_importance_bar_exact(cf_shap[i], 0.5))
        ], style=row_style))

    alert_div = "Alert prediction probability : {:.4f}".format(get_prediction(alert)[0])
    cf_div = "Counterfactual prediction probability : {:.4f}".format(get_prediction(cf)[0])

    return [table, alert_div, cf_div]


if __name__ == '__main__':
    app.run_server(debug=False)
