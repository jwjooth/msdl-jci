import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# read data csv
df = pd.read_csv("../data/raw/bi_rate.csv")

# data cleaning
df["BI-7Day-RR"] = df["BI-7Day-RR"].astype(str).str.replace("%", "").str.strip().astype(float)

# take the close price column
data = df["BI-7Day-RR"].values.reshape(-1, 1)

# normalized data scale 0 to 1, easy to process in neural network purposes
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# create function to change time-series data become "learning from past" pattern
# example: using 3 months ago to predict the fourth month
def create_dataset(data_set, look_back: int):
    X, y = [], []
    for i in range(len(data_set) - look_back):
        X.append(data_set[i:(i + look_back), 0])
        y.append(data_set[i + look_back, 0])
    return np.array(X), np.array(y)


# total months ago to become a target
target_look_back = 3

X, y = create_dataset(scaled_data, target_look_back)

# split the training data (80%) and testing data (20%)
# for pure time-series, usually without shuffle, for this case we used simple split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# create and train MLP model
# initialize MLPRegressor
# hidden_layer_size=(50, 50) mean 2 hidden layer with 50 neuron each
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    max_iter=500,
    activation='relu',
    solver='adam',
    random_state=42,
)

print("Training MLP model...")

mlp_model.fit(X_train, y_train)

print("training done!")

# evaluate and prediction
# do prediction with testing data
prediction_scaled = mlp_model.predict(X_test)

# denormalized prediction data to currency value
prediction_scaled = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# calculate error (mse, mae, mape)
mse = mean_squared_error(y_test_actual, prediction_scaled)
mae = mean_absolute_error(y_test_actual, prediction_scaled)
mape = mean_absolute_percentage_error(y_test_actual, prediction_scaled)

print(f"""
mean squared error: {mse:.2f}
mean absolute error: {mae:.2f}
mean absolute percentage error: {mape:.2f}
""")

# show 5 examples prediction data vs the real value
print("\n-- prediction result usd/idr examples --")

for i in range(18):
    print(f"prediction: {prediction_scaled[i][0]:.2f}% | real: {y_test_actual[i][0]:.2f}%")
