import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the model
model = tf.keras.models.load_model('best_model.h5')

# Load stats and results data
stats_df = pd.read_csv('stats.csv')
results_df = pd.read_csv('results.csv')

stats_df['team'] = stats_df['team'].str.replace(" ", "")
results_df['home_team'] = results_df['home_team'].str.replace(" ", "")
results_df['away_team'] = results_df['away_team'].str.replace(" ", "")

# Initialize and fit OneHotEncoder for team names
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(results_df[['home_team', 'away_team']])

# Prepare and fit StandardScaler for features
# Using only the statistical features for fitting the scaler
stats_features = ['wins', 'losses', 'goals']
scaler = StandardScaler()
scaler.fit(stats_df[stats_features])

def predict_match_outcome_with_probability(home_team, away_team):
    if not home_team or not away_team:
        raise ValueError("No data provided for prediction")   
    # Encode the input teams
    encoded_input = encoder.transform(np.array([[home_team], [away_team]]).T).toarray()

    # Flatten the encoded input to make it one-dimensional
    encoded_input_flattened = encoded_input.flatten()

    # Fetch and flatten the stats for both teams
    home_stats = scaler.transform(stats_df[stats_df['team'] == home_team][stats_features].iloc[-1:].values).flatten()
    away_stats = scaler.transform(stats_df[stats_df['team'] == away_team][stats_features].iloc[-1:].values).flatten()

    # Concatenate the flattened arrays
    X_new = np.concatenate([encoded_input_flattened, home_stats, away_stats])

    # Reshape X_new to match the input shape the model expects
    X_new = X_new.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(X_new)
    outcome = np.argmax(prediction, axis=1)
    probability = np.max(prediction, axis=1)

    # Mapping the prediction to the outcome
    outcome_map = {0: 'Home win', 1: 'Draw', 2: 'Away win'}
    predicted_outcome = outcome_map[outcome[0]]
    predicted_probability = probability[0]

    return predicted_outcome, float("{:.2f}".format(predicted_probability))

outcome, probability = predict_match_outcome_with_probability('Liverpool', 'LeicesterCity')
print(f'Predicted Outcome: {outcome}, Probability: {probability:.2f}')