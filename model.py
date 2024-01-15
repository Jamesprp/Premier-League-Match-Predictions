import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

encoder = OneHotEncoder(handle_unknown='ignore')

# Load data
results_df = pd.read_csv('results.csv')
stats_df = pd.read_csv('stats.csv')



# Preprocess team names

  # Replace 'team' with actual column name

# Basic Preprocessing and Feature Engineering
# For demonstration, let's take basic stats like wins, losses, and goals
stats_df = stats_df[['team', 'wins', 'losses', 'goals', 'season']]

# Merging stats with results
merged_df = results_df.merge(stats_df, left_on=['home_team', 'season'], right_on=['team', 'season'])
merged_df = merged_df.merge(stats_df, left_on=['away_team', 'season'], right_on=['team', 'season'], suffixes=('_home', '_away'))
# Dropping unnecessary columns
merged_df.drop(['team_home', 'team_away'], axis=1, inplace=True)

# Encode categorical variables
encoder = OneHotEncoder()
encoded_teams = encoder.fit_transform(merged_df[['home_team', 'away_team']]).toarray()

# Prepare X and Y
X = pd.concat([pd.DataFrame(encoded_teams), merged_df[['wins_home', 'losses_home', 'goals_home', 'wins_away', 'losses_away', 'goals_away']]], axis=1)
y = merged_df['result'].map({'H': 0, 'D': 1, 'A': 2})  # Home win, Draw, Away win

# Convert all column names to string
X.columns = X.columns.astype(str)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert y to categorical
y = to_categorical(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)


model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X, y):
    # Define the model architecture
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.5))  # Adding dropout layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 output classes

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Generate a print
    print(f'Training for fold {fold_no} ...')

    # Fit data to model with ModelCheckpoint callback
    model.fit(X[train], y[train], epochs=50, batch_size=32, validation_data=(X[test], y[test]), callbacks=[model_checkpoint])

    fold_no += 1
    best_model = load_model('best_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

def predict_match_outcome_with_probability(home_team, away_team):
    # Encode the input teams
    encoded_input = encoder.transform([[home_team, away_team]]).toarray()
    
    # Create a DataFrame with the correct column names for the encoded data
    encoded_columns = encoder.get_feature_names_out(input_features=['home_team', 'away_team'])
    encoded_df = pd.DataFrame(encoded_input, columns=encoded_columns)

    # Fetch the stats for both teams
    home_stats = stats_df[stats_df['team'] == home_team].iloc[-1][['wins', 'losses', 'goals']]
    away_stats = stats_df[stats_df['team'] == away_team].iloc[-1][['wins', 'losses', 'goals']]

    # Prepare the input features
    stats_columns = ['wins', 'losses', 'goals']
    X_new = pd.concat([encoded_df.reset_index(drop=True), pd.DataFrame([home_stats.values, away_stats.values], columns=stats_columns).reset_index(drop=True)], axis=1)

    # Add missing columns with zeros
    missing_cols = set(scaler.feature_names_in_) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0

    # Ensure columns order matches the scaler's expectation
    X_new = X_new[scaler.feature_names_in_]

    # Normalize the features
    X_new = scaler.transform(X_new)

    # Make a prediction and get probabilities
    prediction = model.predict(X_new)
    outcome = np.argmax(prediction, axis=1)
    probability = np.max(prediction, axis=1)

    # Map the prediction to the outcome
    outcome_map = {0: 'Home win', 1: 'Draw', 2: 'Away win'}
    predicted_outcome = outcome_map[outcome[0]]
    predicted_probability = probability[0]

    return predicted_outcome, predicted_probability

# Example usage
outcome, probability = predict_match_outcome_with_probability('Liverpool','Leicester City')
print(f'Predicted Outcome: {outcome}, Probability: {probability:.2f}')


