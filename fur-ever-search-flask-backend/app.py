from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Get a list of all dog breeds
@app.route('/breeds', methods=['GET'])
def get_breeds():
    breeds_list = rawdata['ï»¿breed'].tolist()
    return jsonify({'breeds': breeds_list})

# Get recommendations for a specific breed
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    breed_name = request.args.get('breed_name')
    recommendations = get_recommendations(breed_name)  # Implement this function using your logic
    return jsonify({'recommendations': recommendations.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
