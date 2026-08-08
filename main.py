import pandas as pd

from models.macro.mlp_bi_rate import predict_bi_rate

data = pd.read_csv("./data/raw/bi_rate.csv")

prediction_data, actual_data, error = predict_bi_rate(data)

print("mse:", error["mse"])
print("mae:", error["mae"])
print("mape:", error["mape"])

for i in range (len(actual_data)):
    print(f"prediction: {prediction_data[i][0]:.2f}% || actual data: {actual_data[i][0]:.2f}%")