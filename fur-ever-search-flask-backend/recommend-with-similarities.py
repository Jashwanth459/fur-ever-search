import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import pandas as pd
import numpy as np


def read_data(feature_list):
    rawdata = pd.read_csv('dog_breeds_dataset.csv', encoding='unicode_escape')
    rawdata = rawdata.fillna(0);
    # feature_list = ['Affectionate With Family', 'Good With Young Children', 'Good With Other Dogs']
    # feature_list = ['Shedding Level', 'Coat Grooming Frequency', 'Drooling Level', 'Coat Type', 'Coat Length']
    # feature_list = ['Openness To Strangers', 'Playfulness Level', 'Watchdog/Protective Nature', 'Adaptability Level']
    # feature_list = ['Trainability Level', 'Energy Level', 'Barking Level', 'Mental Stimulation Needs']
    if feature_list:
        features = ['temperament'] + feature_list
    else:
        features = ['temperament', 'min_height', 'max_height', 'min_weight', 'max_weight',
                    'min_expectancy', 'max_expectancy', 'Affectionate With Family',
                    'Good With Young Children', 'Good With Other Dogs', 'Shedding Level',
                    'Coat Grooming Frequency', 'Drooling Level', 'Coat Type', 'Coat Length',
                    'Openness To Strangers', 'Playfulness Level', 'Watchdog/Protective Nature',
                    'Adaptability Level', 'Trainability Level', 'Energy Level', 'Barking Level',
                    'Mental Stimulation Needs']

    data = rawdata[features].copy()

    # Temporarily dropping other hot encoded columns
    encoding_list = ['Coat Type', 'Coat Length']
    result = any(feature in encoding_list for feature in features)

    # print('result - jashp', result)
    # print('features - jashp', features)

    if result:
        data.drop(['temperament', 'Coat Type', 'Coat Length'], axis=1, inplace=True)
    else:
        data.drop(['temperament'], axis=1, inplace=True)

    if data.empty:
        only_coat = True
        columns_before = 0
        cosine_sim = 0
    else:
        only_coat = False
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        X = scaled_data.copy()
        columns_before = len(data.columns.tolist())
        cosine_sim = cosine_similarity(X, X)
   

    # standardization of all the numerical columns
    # object = StandardScaler()
    # scaled_data = object.fit_transform(data)
    # X = scaled_data.copy()
    # columns_before = len(data.columns.tolist())

    # encoding of temperament
    data['temperament list'] = rawdata['temperament'].apply(lambda x: x.lower().split(',') if type(x) == str else [])
    temperament = []
    for i in data['temperament list']:
        temperament.extend(i)
    temperament_no_repeats = set(temperament)
    data['one-hot temperament'] = data['temperament list'].apply(
        lambda x: [int(temperament in x) for temperament in temperament_no_repeats])

    # encoding of group
    rawdata['group'] = rawdata['group'].fillna('none')
    group_no_repeats = rawdata['group'].unique()
    data['one-hot group'] = rawdata['group'].apply(lambda x: [int(group in x) for group in group_no_repeats])

    if result:
        # encoding of  coat type 'Coat Type', 'Coat Length'
        data['coat type list'] = rawdata['Coat Type'].apply(lambda x: x.lower().split(',') if type(x) == str else [])
        coat_type = []
        for i in data['coat type list']:
            coat_type.extend(i)
        coat_type_no_repeats = set(coat_type)
        data['one-hot coat type'] = data['coat type list'].apply(
            lambda x: [int(coat_type in x) for coat_type in coat_type_no_repeats])

        # encoding of coat type 'Coat Type', 'Coat Length'
        data['coat length list'] = rawdata['Coat Length'].apply(
            lambda x: x.lower().split(',') if type(x) == str else [])
        coat_length = []
        for i in data['coat length list']:
            coat_length.extend(i)
        coat_length_no_repeats = set(coat_length)
        data['one-hot coat length'] = data['coat length list'].apply(
            lambda x: [int(coat_length in x) for coat_length in coat_length_no_repeats])
        # create sims array
        sims = np.zeros([len(data), len(data)])
        # calculate cosine of similarity of encoded groups
        for col in ['one-hot temperament', 'one-hot group', 'one-hot coat type', 'one-hot coat length']:
            sims += np.array(cosine_similarity(data[col].tolist(), data[col].tolist()))
    else:
        # create sims array
        sims = np.zeros([len(data), len(data)])
        # calculate cosine of similarity of encoded groups
        for col in ['one-hot temperament', 'one-hot group']:
            sims += np.array(cosine_similarity(data[col].tolist(), data[col].tolist()))

   # standardization of all the numerical columns
    if only_coat:
        similarity_scores = sims
    else:
        sims = sims / columns_before
        similarity_scores = cosine_sim + sims
    return rawdata, similarity_scores


# get recommendations based on cosine similarity scores
def get_recommendations(breed_name, feature_list):
    rawdata, cosine_sim = read_data(feature_list)
    idx = rawdata[rawdata['ï»¿breed'] == breed_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    breed_indices = [i[0] for i in sim_scores[1:10]]  # Get top 5 similar breeds
    return rawdata['ï»¿breed'].iloc[breed_indices]

# Example usage: Get recommendations for 'Affenpinscher'
# recommendations = get_recommendations('Poodle (Toy)')
# print(recommendations)

def get_breed_info(breed_name):
    rawdata, cosine_sim = read_data([])
    # print('rawdata', rawdata)
    return rawdata[rawdata['ï»¿breed'] == breed_name]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

# Get a list of all dog breeds
@app.route('/breeds', methods=['GET'])
def breeds():
    rawdata, cosine_sim = read_data([])
    # print('rawdata - jashp', rawdata.head(5))
    breeds_list = rawdata.to_dict(orient='records')
    # print('breeds_list', breeds_list[0])
    return jsonify({'breeds': breeds_list})

# Get info of a specific breed
@app.route('/breeds/<string:breed_name>', methods=['GET'])
def breed_info(breed_name):
    breed_info = get_breed_info(breed_name).to_dict(orient='records')

    return jsonify({'breed_info': breed_info})

# Get recommendations for a specific breed
@app.route('/recommendations/<string:breed_name>', methods=['POST'])
def recommendations_for_breed(breed_name):
    # print('breed_name is jashp', breed_name)
    if request.is_json:
        data = request.get_json()
        feature_list = data.get('recommendationPreferences')
        # print('recommendation_preferences ', feature_list)
        recommendations = get_recommendations(breed_name, feature_list)
        if recommendations.empty:
            return jsonify({'error': 'Breed not found'}), 404
        recommendations_list = recommendations.tolist()
        return jsonify({'recommendations': recommendations_list})
    else:
        return jsonify({'error': 'Invalid request format. Must be JSON'}), 400

if __name__ == '__main__':
    app.run(debug=True)

