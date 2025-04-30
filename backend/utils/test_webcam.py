import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"E:\project\backend\model\sign_language_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load LabelEncoder
label_encoder = joblib.load(r"E:\project\backend\model\label_encoder.pkl")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to normalize and extract coordinates for both hands
def preprocess_two_hands(multi_hand_landmarks, handedness):
    left_hand = None
    right_hand = None

    for idx, hand_label in enumerate(handedness):
        label = hand_label.classification[0].label
        if label == 'Left':
            left_hand = multi_hand_landmarks[idx]
        else:
            right_hand = multi_hand_landmarks[idx]

def extract_landmarks_raw(landmarks):
    return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

def preprocess_two_hands(multi_hand_landmarks, handedness):
    left_hand = None
    right_hand = None

    for idx, hand_label in enumerate(handedness):
        label = hand_label.classification[0].label
        if label == 'Left':
            left_hand = multi_hand_landmarks[idx]
        else:
            right_hand = multi_hand_landmarks[idx]

    coords = []

    # Left hand
    if left_hand:
        coords.extend(np.array(extract_landmarks_raw(left_hand)).flatten())
    else:
        coords.extend([0.0] * 63)

    # Right hand
    if right_hand:
        coords.extend(np.array(extract_landmarks_raw(right_hand)).flatten())
    else:
        coords.extend([0.0] * 63)

    return coords



# Start Webcam
cap = cv2.VideoCapture(0)
print("📸 Starting camera... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    coords = []

    if result.multi_hand_landmarks and result.multi_handedness:
        coords = preprocess_two_hands(result.multi_hand_landmarks, result.multi_handedness)

        # Draw landmarks
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Predict only if we have at least one hand
    if coords:
        print(f"\n📤 Landmarks sent to model ({len(coords)} values):\n{coords}\n")
        input_data = np.array(coords, dtype=np.float32).reshape(1, -1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Sign Language Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
