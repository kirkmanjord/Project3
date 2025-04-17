import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dropout

matplotlib.use("TkAgg")     # or "TkAgg" / "MacOSX" depending on your OS
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam


np.set_printoptions(precision=3, suppress=True, floatmode='maxprec_equal')
# First, split into train+val and test (e.g., 80/20)


df = pd.read_pickle("dataZillow.pkl")
df = df.sample(frac=1).reset_index(drop=True)
df.drop(columns = [df.columns[0],df.columns[1]], inplace=True)



#df.drop(columns= ['RegionID','SizeRank'], inplace=True)
ys = df.iloc[:, 157]

df.drop(columns = [df.columns[157]] , inplace=True)
yScaler = StandardScaler()
scaler = StandardScaler()
ys =yScaler.fit_transform(ys.to_frame())
print(f"yerp{yScaler.scale_[0]*0.0086}")
scaled_array = scaler.fit_transform(df)
# Convert back to DataFrame, preserving original columns and index
df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
#X_temp, X_test, y_temp, y_test = train_test_split(df.drop(columns = []), df, test_size=0.2, random_state=42)
count = 0
for col in df.columns:
    print(f'{col} is {count}')
    count+=1

print(f"heyyy {yScaler.scale_[0]}")
def getIntoRNNFormat(df):
    date_cols = df.iloc[:,2:157]
    staticFeatures = df.drop(columns = date_cols.columns)
    datas = []
    for idx, row in date_cols.iterrows():
       timeCol = np.array(row).reshape(-1, 1)
       staticMatrix = np.array(staticFeatures.iloc[idx,:]).reshape(1,-1).repeat(timeCol.shape[0], axis = 0)
       final = np.concatenate((timeCol, staticMatrix), axis=1)
       final = timeCol
       datas.append(final)
    return np.stack(datas)



data = getIntoRNNFormat(df)
X_train, X_test, y_train, y_test = train_test_split(data, ys, test_size=0.2)


model = Sequential(
    [
        SimpleRNN(
            units=20,
            activation="tanh",
            input_shape=(X_train.shape[1], X_train.shape[2]),

        ),
        #Dropout(0.5),
        Dense(1),
    ]
)

model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
model.summary()

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=40,
    verbose=2,
)

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.4f}")

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(np.array(history.history["mae"]) * yScaler.scale_[0], label="Train Loss")
plt.plot(np.array(history.history["val_mae"]) * yScaler.scale_[0], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("RNN Training Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Try to show the plot. If PyCharm backend cannot display, save to file.
try:
    plt.show()
except Exception as e:
    fallback_path = "training_loss.png"
    plt.savefig(fallback_path)
    print(
        f"Plot backend unavailable ({e}). Figure saved to '{fallback_path}'."
    )
y_pred_scaled = model.predict(X_test)

# Inverse transform both y_test and y_pred to original units
y_pred = yScaler.inverse_transform(y_pred_scaled)
y_true = yScaler.inverse_transform(y_test)

# Flatten for plotting
y_pred = y_pred.flatten()
y_true = y_true.flatten()

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(y_true, label="True Prices", linewidth=2)
plt.plot(y_pred, label="Predicted Prices", linestyle='--')
plt.title("House Price Prediction: True vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Get MSE from training history
train_mse = np.array(history.history["loss"])
val_mse = np.array(history.history["val_loss"])

# Convert to original units (since training was on scaled y)
# MSE scales with the square of standard deviation
scale_factor = yScaler.scale_[0] ** 2
train_mse_orig = train_mse * scale_factor
val_mse_orig = val_mse * scale_factor

# RMSE = sqrt(MSE)
train_rmse = np.sqrt(train_mse_orig)
val_rmse = np.sqrt(val_mse_orig)

# Compute R² from MSE manually using: R² = 1 - (MSE / Var(y_true))
# Use variance of the original unscaled training and test targets
y_train_true = yScaler.inverse_transform(y_train)
y_test_true = yScaler.inverse_transform(y_test)

y_train_var = np.var(y_train_true)
y_test_var = np.var(y_test_true)

train_r2 = 1 - (train_mse_orig / y_train_var)
val_r2 = 1 - (val_mse_orig / y_test_var)

# Plot RMSE
plt.figure(figsize=(10, 4))
plt.plot(train_rmse, label="Train RMSE")
plt.plot(val_rmse, label="Val RMSE")
plt.title("RMSE Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("RMSE (original units)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot R²
plt.figure(figsize=(10, 4))
plt.plot(train_r2, label="Train R²")
plt.plot(val_r2, label="Val R²")
plt.title("R² Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()