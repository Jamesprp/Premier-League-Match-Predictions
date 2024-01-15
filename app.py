from flask import Flask, request, jsonify, render_template
from functionpredictor import predict_match_outcome_with_probability
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        home_team = request.json['team1']
        away_team = request.json['team2']
        outcome, probability = predict_match_outcome_with_probability(home_team, away_team)
        return jsonify({'outcome': outcome, 'probability': probability})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=5501)
