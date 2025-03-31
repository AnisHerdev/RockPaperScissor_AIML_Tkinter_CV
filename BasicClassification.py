import cv2 as cv
import numpy as np
import csv
import copy
import itertools
import argparse
from model import KeyPointClassifier
import mediapipe as mp
from utils import CvFpsCalc

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def preprocess_and_classify(image, keypoint_classifier):
    """
    Preprocess the image and classify it using the KeyPointClassifier.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        keypoint_classifier (KeyPointClassifier): The trained keypoint classifier.

    Returns:
        str: The classification result ("rock", "paper", "scissors", or "unknown").
    """
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Convert the image to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process the image to detect hand landmarks
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmark_list = calc_landmark_list(image, hand_landmarks)

            # Preprocess the landmarks
            preprocessed_landmarks = pre_process_landmark(landmark_list)

            # Classify the hand gesture
            hand_sign_id = keypoint_classifier(preprocessed_landmarks)

            # Map the classification ID to a label
            labels = ["rock", "paper", "scissors", "unknown"]
            return labels[hand_sign_id] if hand_sign_id < len(labels) else "unknown"

    return "unknown"  # Return "unknown" if no hand is detected


def calc_landmark_list(image, landmarks):
    """
    Calculate the landmark list from Mediapipe landmarks.

    Args:
        image (numpy.ndarray): The input image.
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Hand landmarks.

    Returns:
        list: A list of (x, y) coordinates for the landmarks.
    """
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """
    Preprocess the landmark list for classification.

    Args:
        landmark_list (list): A list of (x, y) coordinates for the landmarks.

    Returns:
        list: A normalized and flattened list of landmark coordinates.
    """
    temp_landmark_list = landmark_list.copy()

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # Flatten the list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalize the coordinates
    max_value = max(map(abs, temp_landmark_list))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,  # Only detect one hand for simplicity
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Hand gesture classification ##########################################################
        hand_sign_text = "No Hand Detected"
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocess landmarks for classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classify the hand gesture
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                hand_sign_text = keypoint_classifier_labels[hand_sign_id]

        # Display classification result ########################################################
        print(f"Gesture: {hand_sign_text}")
        cv.putText(debug_image, f"Gesture: {hand_sign_text}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    print("Program terminated.")