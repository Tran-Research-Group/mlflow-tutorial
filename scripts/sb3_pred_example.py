import mlflow
import numpy as np

mlflow.set_tracking_uri("http://0.0.0.0:5000")

logged_model = "runs:/4125e0adeacf463ca1841cf7e42c9688/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

print("Model loaded. The model object is:", loaded_model)

action, _ = loaded_model.predict(np.array([0.5, 0.5, 0.5]))
print("The model predicts the action:", action)

pass
