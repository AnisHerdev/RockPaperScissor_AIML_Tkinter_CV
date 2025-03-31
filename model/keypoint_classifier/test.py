import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model\keypoint_classifier\keypoint_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def get_keypoints(image):
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Process the image and find hands
            results = hands.process(image_rgb)
            
            # If hands are detected, extract the keypoints
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract keypoints as a list of (x, y) coordinates
                    keypoints = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                    return np.array(keypoints, dtype=np.float32)
        
        # Return an empty array if no hands are detected
        return np.zeros((21, 2), dtype=np.float32)
# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, convert to grayscale, etc.)
    # Replace with your actual preprocessing steps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ... your keypoint detection logic here ... 
    # Assuming you have a function 'get_keypoints' that detects keypoints and returns a numpy array
    keypoints = get_keypoints(gray) 

    # Flatten the keypoints into a 1D array
    keypoints_flattened = keypoints.flatten()

    # Make prediction with the TFLite model
    interpreter.set_tensor(input_details[0]['index'], np.array([keypoints_flattened]))
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(np.squeeze(tflite_results))

    # Display the predicted class on the frame
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Live Video', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()