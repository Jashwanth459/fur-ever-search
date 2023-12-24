from faker import Faker
import pandas as pd
import numpy as np, numpy.random
from faker.providers import DynamicProvider
from datetime import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import warnings
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# Generate synthetic data using Faker and probabilities fom US Census
def load_synthetic_user_data(num=1, seed=None):
    np.random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)
    home_type_provider = DynamicProvider(
        provider_name="home_type",
        elements=["Apartment", "House", "MobileHome"],
    )
    fake.add_provider(home_type_provider)
    today = datetime.date(datetime.now())
    output = []
    us_gender_prob = [0.505, 0.495]
    marital_status_prob = [0.413, 0.587]

    for x in range(num):
        gender = np.random.choice(["M", "F"], p=us_gender_prob)
        marital_status = np.random.choice(["Single", "Married"], p=marital_status_prob)
        if marital_status == "Single":
            family_size = np.random.choice(np.arange(1, 7), p=[0.45, 0.2, 0.15, 0.13, 0.05, 0.02])
        else:
            family_size = np.random.choice(np.arange(2, 7), p=[0.3, 0.2, 0.3, 0.15, 0.05])

        if family_size == 1:
            children = "False"
        else:
            children = np.random.choice(["True", "False"], p=[0.7, 0.3])

        if 3 <= family_size < 5 and children == "True":
            age = today.year - fake.date_of_birth(minimum_age=25, maximum_age=85).year
        elif family_size >= 5 and children == "True":
            age = today.year - fake.date_of_birth(minimum_age=35, maximum_age=85).year
        else:
            age = today.year - fake.date_of_birth(minimum_age=18, maximum_age=85).year

        output.append(
            {
                "user_id": fake.uuid4(),
                "firstname": fake.first_name_male() if gender == "M" else fake.first_name_female(),
                "lastname": fake.last_name(),
                "gender": gender,
                "marital_status": marital_status,
                "zipcode": fake.zipcode(),
                "home_type": fake.home_type(),
                "children": children,
                "family_size": family_size,
                "age": age,
                "user_type": "Synthetic"
            })
    return output


def load_synthetic_user_interaction_data(data_df, dog_df):
    # dog probabilities
    user_list = data_df['user_id'].values.tolist()
    pop_dog_df = dog_df[['akc_breed_popularity', 'ï»¿breed']].sort_values('akc_breed_popularity').dropna()
    # one_to_10 = np.random.normal(1, 0.2, 10)
    # one_to_10 = (one_to_10 / np.sum(one_to_10)) * 0.5
    # remaining = np.random.normal(1, 0.2, len(pop_dog_df) - 10)
    # remaining = (remaining / np.sum(remaining)) * 0.5
    # final = np.sort(np.append(one_to_10, remaining))[::-1]
    final = np.random.normal(1, 0.2, len(pop_dog_df))
    final = final/np.sum(final)
    output = []
    interaction_type = ['Liked', 'ManualSearch', 'Nearbydogs', 'Clicked', 'Video']
    for user in user_list:
        num = np.random.choice(np.arange(1, 7))
        for row in range(num):
            output.append({
                "user_id": user,
                "dog_breed": np.random.choice(pop_dog_df['ï»¿breed'].values.tolist(), p=final),
                "interaction_type": np.random.choice(interaction_type)
            })
    return output


def load_synthetic_data():
    conn = sqlite3.connect('dog_database')
    dog_df = pd.read_csv('dog_breeds_dataset.csv', encoding='unicode_escape')
    # call load_synthetic_user_data user data to generate user data
    user_data = pd.DataFrame(load_synthetic_user_data(num=10000))
    user_data.to_sql('user_data', conn, if_exists='replace', index=False)
    print("Synthetic User data load complete")
    # call load_synthetic_user_interaction_data to generate user interactions data
    user_interaction_data = pd.DataFrame(load_synthetic_user_interaction_data(user_data, dog_df))
    user_interaction_data.to_sql('user_interaction_data', conn, if_exists='replace', index=False)
    print("Synthetic User interaction data load complete")
    conn.commit()
    conn.close()


def insert_actual_user_data(user_id, firstname, lastname, gender, marital_status, zipcode, home_type,
                            children, family_size, age, user_type=None):
    conn = sqlite3.connect('dog_database')
    output = []
    output.append(
        {
            "user_id": user_id,
            "firstname": firstname,
            "lastname": lastname,
            "gender": gender,
            "marital_status": marital_status,
            "zipcode": zipcode,
            "home_type": home_type,
            "children": children,
            "family_size": family_size,
            "age": age,
            "user_type": "Real",
            "user_cluster": "None"
        })
    user_data = pd.DataFrame(output)
    user_data.to_sql('user_data', conn, if_exists='append', index=False)
    print(" Real User data load complete")

    # Real data
    predicted_user_cluster = predict_livedata(user_data)
    top10dogs = get_top10dogs(predicted_user_cluster[0])
    return top10dogs


def actual_insert_user_interaction_data(user_id, dog_breed, interaction_type):
    conn = sqlite3.connect('dog_database')
    output = []
    output.append(
        {
            "user_id": user_id,
            "dog_breed": dog_breed,
            "interaction_type": interaction_type
        })
    user_interaction_data = pd.DataFrame(output)
    user_interaction_data.to_sql('user_interaction_data', conn, if_exists='append', index=False)
    print("Real User interaction data load complete")


def test_data():
    actual_insert_user_data("asdfg12345", "Tony", "Romo", "Male", "Married",
                            75052, "House", "True", 5, 45)
    actual_insert_user_interaction_data("asdfg12345", "Golden Retriever", "Video")
    actual_insert_user_interaction_data("asdfg12345", "Golden Retriever", "Clicked")
    actual_insert_user_interaction_data("asdfg12345", "Golden Retriever", "Liked")
    actual_insert_user_interaction_data("asdfg12345", "Golden Retriever", "ManualSearch")
    actual_insert_user_interaction_data("asdfg12345", "Labrador Retriever", "Liked")


def visualize_cluster(final_data):
    # Exploratory Data Analysis
    # # Checking for collinearity using heat maps
    # corr = final_data.corr()
    # plt.figure(figsize=(16, 6))
    # heatmap = sns.heatmap(final_data.corr(), vmin=-1, vmax=1, annot=True)
    # heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
    # plt.show()

    # Dendogram
    Z = linkage(final_data.corr(), 'average')
    plt.figure(figsize=(35, 20))
    labelsize = 15
    ticksize = 8
    plt.title('Hierarchical Clustering Dendrogram for User Demographics', fontsize=labelsize)
    plt.xlabel('distance', fontsize=20)
    # plt.ylabel('distance', fontsize=20)
    dendrogram(
        Z,
        # rotates the x-axis labels
        # leaf_rotation=90,
        leaf_font_size=10,
        # font size for the x-axis labels
        labels=final_data.columns,
        orientation='left'
    )
    pylab.yticks(fontsize=ticksize)
    pylab.xticks(fontsize=ticksize)
    # plt.show()

    # elbow method
    model = KMeans(n_init=25)
    visualizer = KElbowVisualizer(model, k=(2, 12))
    visualizer.fit(final_data)  # Fit the data to the visualizer
    # visualizer.show()  # Draw/show/show the data

    # calinski_harabasz
    visualizer = KElbowVisualizer(model, k=(2, 15), metric='calinski_harabasz', locate_elbow=False, timings=False)
    visualizer.fit(final_data)  # Fit the data to the visualizer
    # visualizer.show()  # Draw/show/show the data

    # Silhouette score
    range_n_clusters = range(2, 10)
    for n_clusters in range_n_clusters:
        # Initializing the model with n_clusters value and a random   generator
        model = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = model.fit_predict(final_data)
        # The silhouette_score gives the average value for all the   samples.
        # Calculating number of clusters
        silhouette_avg = silhouette_score(final_data, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhoutte_score is :", silhouette_avg)
        # Using Silhouette Plot
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        # Fit the data to the visualizer
        visualizer.fit(final_data)
        # Render the figure
        # visualizer.show()

        # pca
        # pca = PCA(2)
        # pca_data = pd.DataFrame(pca.fit_transform(final_data), columns=['PC1', 'PC2'])
        # model = KMeans(n_clusters=n_clusters)
        # pca_data['cluster'] = model.fit_predict(final_data)
        # sns.scatterplot(x="PC1", y="PC2", hue="cluster", palette=sns.color_palette("hls", 10), data=pca_data,
        #                 legend="full")
        # plt.show()


def transform_data(user_data_df, process_type=None):
    categorical_data = user_data_df[['gender', 'marital_status', 'home_type', 'children']].astype(str)
    continuous_data = user_data_df[['family_size', 'age']]
    if process_type == 'Train':
        one_hot_encoder = OneHotEncoder(sparse=False)
        encoded_data = one_hot_encoder.fit_transform(categorical_data).astype(float)
        pickle.dump(one_hot_encoder, open("encoder.pkl", 'wb'))

        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(continuous_data).astype(float))
        pickle.dump(scaler, open("min_max_scaler.pkl", 'wb'))
    else:
        one_hot_encoder = pickle.load(open("encoder.pkl", 'rb'))
        encoded_data = one_hot_encoder.transform(categorical_data).astype(float)

        scaler = pickle.load(open("min_max_scaler.pkl", 'rb'))
        scaled_data = pd.DataFrame(scaler.transform(continuous_data).astype(float))

    categorical_data = pd.DataFrame(encoded_data)
    categorical_data.columns = one_hot_encoder.get_feature_names_out()
    scaled_data.columns = continuous_data.columns
    final_data = pd.concat([categorical_data, scaled_data], axis=1)
    return final_data


def load_user_cluster():
    conn = sqlite3.connect('dog_database')
    user_data_df = pd.read_sql_query('select * from user_data', conn)
    final_data = transform_data(user_data_df, 'Train')

    # visualize_cluster(final_data)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=6, n_init=25, random_state=42)
    kmeans.fit(final_data)

    # Add cluster labels to the original dataset
    user_data_df['user_cluster'] = final_data['cluster_label'] = kmeans.labels_
    # sns.countplot(final_data, x="cluster_label")
    # plt.show()

    user_data_df.to_sql('user_data', conn, if_exists='replace', index=False)
    print("Cluster data load complete")


def load_interaction_score():
    interact_type = ['Liked', 'ManualSearch', 'Nearbydogs', 'Clicked', 'Video']
    scores = [5, 3, 4, 1, 2]
    scores = {
        'interaction_type': interact_type,
        'interaction_score': scores
    }
    interaction_score = pd.DataFrame(scores)
    conn = sqlite3.connect('dog_database')
    user_data_df = pd.read_sql_query('select * from user_data', conn)
    user_interaction_data_df = pd.read_sql_query('select * from user_interaction_data', conn)
    user_data_cluster = pd.merge(user_interaction_data_df,
                                 user_data_df[['user_id', 'user_cluster']], on='user_id', how='left')
    combine_scores = pd.merge(user_data_cluster, interaction_score, on='interaction_type', how='left')
    aggregate_scores = combine_scores.groupby(['user_cluster', 'dog_breed'])['interaction_score'].agg(
        'sum').reset_index()
    top10 = (aggregate_scores.groupby('user_cluster').
             apply(lambda x: x.nlargest(10, 'interaction_score')).reset_index(drop=True))
    top10.to_sql('top10dogs', conn, if_exists='replace', index=False)
    print("Top 10 dogs per cluster load complete")


def get_top10dogs(cluster_num):
    conn = sqlite3.connect('dog_database')
    sql_statement = 'select dog_breed from top10dogs where user_cluster = ' + str(cluster_num)
    top10dogs = pd.read_sql(sql_statement, con=conn).values.reshape(-1, ).tolist()
    return top10dogs


def train_random_forests():
    conn = sqlite3.connect('dog_database')
    user_data_df = pd.read_sql_query('select * from user_data', conn)
    X = user_data_df.iloc[:, :-2]
    y = user_data_df["user_cluster"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    X_train = transform_data(X_train, 'Train')
    X_test = transform_data(X_test)

    # # hyperparameter tuning
    # hyper_params = {
    #     'max_depth': range(3, 20),
    #     'max_features': range(2, 4),
    #     'min_samples_leaf': range(20, 400, 50),
    #     'n_estimators': range(10, 101, 10)
    # }

    # model = RandomForestClassifier(class_weight='balanced_subsample')
    # model_rcv = RandomizedSearchCV(estimator=model,
    #                                param_distributions=hyper_params,
    #                                verbose=1,
    #                                cv=5,
    #                                return_train_score=True,
    #                                n_jobs=1,
    #                                n_iter=50)
    # model_rcv.fit(X_train, y_train)
    # print(model_rcv.best_score_)
    # print(model_rcv.best_estimator_)
    # print(type(model_rcv.best_estimator_))

    forest_best_params = RandomForestClassifier(class_weight='balanced_subsample', max_depth=19, max_features=2,
                                                min_samples_leaf=120, n_estimators=30)
    forest_best_params.fit(X_train, y_train)
    print("tset", forest_best_params.feature_importances_)

    # modelname.feature_importance_
    # y_axis = forest_best_params.feature_importances_
    # # plot
    # fig, ax = plt.subplots()
    # width = 0.4  # the width of the bars
    # ind = np.arange(len(y_axis))  # the x locations for the groups
    # ax.barh(ind, y_axis, width, color="green")
    # ax.set_yticks(ind + width / 10)
    # ax.set_yticklabels(col, minor=False)
    # plt.title('Feature importance in RandomForest Classifier')
    # plt.xlabel('Relative importance')
    # plt.ylabel('feature')
    # plt.figure(figsize=(5, 5))
    # fig.set_size_inches(6.5, 4.5, forward=True)

    # # Export as dot file
    # export_graphviz(forest_best_params.estimators_[0], out_file='tree.dot',
    #                 # feature_names=[X_train.columns],
    #                 # class_names=[y_train.columns],
    #                 rounded=True, proportion=False,
    #                 precision=2, filled=True)
    #
    # # Convert to png using system command (requires Graphviz)
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    #
    # # Display in jupyter notebook
    #
    # Image(filename='tree.png')
    #

    pickle.dump(forest_best_params, open("forest_params.pkl", 'wb'))
    y_pred_test = forest_best_params.predict(X_test)

    plt.figure(figsize=(20, 20))
    _ = tree.plot_tree(forest_best_params.estimators_[0], feature_names=X_train.columns.values, filled=True)

    # View accuracy score
    print(accuracy_score(y_test, y_pred_test))

    # View confusion matrix for test data and predictions
    print(confusion_matrix(y_test, y_pred_test))

    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['0', '1', '2', '3', '4', '5']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    # plt.show()

    # View the classification report for test data and predictions
    print(classification_report(y_test, y_pred_test))


def predict_livedata(real_data_df):
    # Real data
    X_real = transform_data(real_data_df)
    forest_best_params = pickle.load(open("forest_params.pkl", 'rb'))
    y_real = forest_best_params.predict(X_real)
    return y_real


# train_random_forests()


# load_synthetic_data()
# load_user_cluster()
# load_interaction_score()
# train_random_forests()

# real_data = [{
#     "user_id": "asdfg12345",
#     "firstname": "Tony",
#     "lastname": "Romo",
#     "gender": "M",
#     "marital_status": "Married",
#     "zipcode": 75052,
#     "home_type": "MobileHome",
#     "children": "False",
#     "family_size": 1,
#     "age": 25,
#     "user_type": "Real",
#     "user_cluster": "None"
# }]
# real_data_df = pd.DataFrame(real_data)
# predicted_user_cluster = predict_livedata(real_data_df)
# top10dogs = get_top10dogs(predicted_user_cluster[0])
# print(top10dogs)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

# Get recommendations for a specific breed
@app.route('/breeds', methods=['POST'])
def recommendations_for_user():
    print('request - jashp', request)
    
    if request.is_json:
        data = request.get_json()
        user_info = data.get('userInfo')
        print('user_info ', user_info)
        real_data_df = pd.DataFrame([user_info])
        predicted_user_cluster = predict_livedata(real_data_df)
        top10dogs = get_top10dogs(predicted_user_cluster[0])
        print('top10dogs', top10dogs)
        return jsonify({'recommendations': top10dogs})
    else:
        return jsonify({'error': 'Invalid request format. Must be JSON'}), 400
    
# Get recommendations for a specific breed
@app.route('/user-interactions', methods=['POST'])
def update_user_interactions():
    print('request - jashp', request)
    
    if request.is_json:
        data = request.get_json()
        user_interaction = data.get('userInteraction')
        print('user_info ', user_interaction)
        uid = user_interaction.get('uid')
        breed_name = user_interaction.get('breed')
        interaction_type = user_interaction.get('interaction_type')
        actual_insert_user_interaction_data(uid, breed_name, interaction_type)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'error': 'Invalid request format. Must be JSON'}), 400
    
# Get recommendations for a specific breed
@app.route('/breedss', methods=['GET'])
def get_breeds():
    return jsonify({'error': 'Invalid request format. Must be JSON'}), 400

# API Implementation start

# API Implementation end

if __name__ == '__main__':
    app.run(debug=True, port=5001)