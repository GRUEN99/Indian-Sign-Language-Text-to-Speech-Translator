# inference.py
import numpy as np
import joblib
import tensorflow as tf

# Load label encoder used during training
label_encoder = joblib.load(r"E:\project\backend\model\label_encoder.pkl") 

# Load the TFLite model once
interpreter = tf.lite.Interpreter(model_path = r"E:\project\backend\model\sign_language_model.tflite")
interpreter.allocate_tensors()

def predict_letter(landmarks: list[float]) -> str:
    if len(landmarks) != 126:
        # Pad to 126 if one hand is missing
        landmarks += [0.0] * (126 - len(landmarks))

    input_data = np.array([landmarks], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    print(f"Received Landmarks: {landmarks}")  # Debug in terminal
    print(f"Predicted Index: {predicted_index}, Label: {predicted_label}")

    return predicted_label
