import mlflow
import numpy as np

mlflow.set_tracking_uri("http://0.0.0.0:5000")

logged_model = "runs:/36f4d7ea41ec40289cdabe9d10e57bb2/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

print("Model loaded. The model object is:", loaded_model)

loaded_model.predict(np.array([0.5, 0.5, 0.5]))

pass
