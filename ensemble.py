from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
#from sklearn.model_selection import train_test_split

def ensemble(data, labels):
    X = data
    y = labels
    
    # Crea un MLPRegressor como clasificador base
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)

    # Crea un BaggingRegressor con el MLPRegressor como base
    bagging_regressor = BaggingRegressor(base_estimator=mlp_regressor, n_estimators=10, random_state=42)

    # Entrena el BaggingRegressor en los datos de entrenamiento
    bagging_regressor.fit(X, y)

    return

# Realiza predicciones en los datos de prueba
#y_pred = bagging_regressor.predict(X_test)

# Evalúa el rendimiento del modelo (por ejemplo, usando el error cuadrado medio)
#mse = mean_squared_error(y_test, y_pred)
#print("Error Cuadrado Medio:", mse)
