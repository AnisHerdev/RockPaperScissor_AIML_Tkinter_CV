import random
import tkinter as tk
# from app import main  
import cv2
from PIL import Image, ImageTk
from PIL.Image import Resampling
from model import KeyPointClassifier  # Import KeyPointClassifier
import mediapipe as mp
# from utils import CvFpsCalc
# from collections import deque, Counter
import itertools
import pygame


class GUI:
    def __init__(self):
        pygame.mixer.init()
        self.countdown_sound = pygame.mixer.Sound("countdown.mp3")
        pygame.mixer.music.load("game-start.mp3")
        self.root = tk.Tk()
        self.root.title("Rock Paper Scissors") # name of the window
        self.root.geometry("800x500")
        self.root.config(bg="light green")
        self.mainLabel = tk.Label(self.root, text="Rock Paper Scissors", font=("Arial", 24))
        self.mainLabel.pack(pady=20)

        self.points = tk.IntVar()
        self.points.set(3) # default value for points to win

        self.getNumpointsFrame = tk.Frame(self.root)
        self.getNumpointsFrame.config(bg="light green")
        self.getNumpointsFrame.columnconfigure(0,weight=1)
        self.getNumpointsFrame.columnconfigure(1,weight=1)
        self.getNumpointsFrame.columnconfigure(2,weight=1)

        self.pointsLabel = tk.Label(self.getNumpointsFrame, text="Points to Win", font=('Arial',16),bg="light green")
        self.pointsLabel.grid(row = 0,column=0, padx=10)
        self.pointsEntry = tk.Entry(self.getNumpointsFrame, textvariable=self.points, font=('Arial',16),width=5)
        self.pointsEntry.grid(row = 0,column=1, padx=10)
        self.pointsToWinBtn = tk.Button(self.getNumpointsFrame, text="Start Game", font=('Arial',16),command=self.pointsToWin)
        self.pointsToWinBtn.grid(row = 0,column=2,padx=10)

        self.getNumpointsFrame.pack(pady=20)

        self.pointsFrame = tk.Frame(self.root)
        self.pointsFrame.config(bg="light green")
        self.pointsFrame.columnconfigure(0,weight=1)
        self.pointsFrame.columnconfigure(1,weight=1)

        self.humanPoints=tk.IntVar()
        self.humanPoints.set(0)
        self.computerPoints=tk.IntVar()
        self.computerPoints.set(0)
        self.humanPointsLabel = tk.Label(self.pointsFrame, text=f"Your Points: {self.humanPoints.get()}", font=('Arial',16),bg="light green")
        self.humanPointsLabel.grid(row = 0,column=0, padx=10)
        self.computerPointsLabel = tk.Label(self.pointsFrame, text=f"Computer Points: {self.computerPoints.get()}", font=('Arial',16),bg="light green")
        self.computerPointsLabel.grid(row = 0,column=1, padx=10)

        self.label = tk.Label(self.root, text="Choose an option", font=("Arial", 16))

        # Video Frame
        self.videoFrame = tk.Frame(self.root)
        self.videoFrame.columnconfigure(0, weight=1)
        self.videoFrame.columnconfigure(1, weight=1)
        self.canvas = tk.Canvas(self.videoFrame, width=620, height=480) 
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.computerChoiceImg = tk.Label(self.videoFrame)
        self.computerChoiceImg.grid(row=0, column=1, padx=10, pady=10)

        static_image = Image.open("scissor.jpg").resize((620, 480), Image.Resampling.LANCZOS)
        static_image_tk = ImageTk.PhotoImage(static_image)
        self.computerChoiceImg.config(image=static_image_tk)
        self.computerChoiceImg.image = static_image_tk  # Keep a reference to avoid garbage collection
        # self.btn = tk.Button(self.root, text="Computer", font=('Arial', 16), command=self.updateComputerChoice) # UpdatecomputerChoice manually
        # self.btn.pack()
        self.btn2 = tk.Button(self.root, text="ScoreUpdate", font=('Arial', 16), command=self.updateScore) # UpdateScore manually
        self.btn2.pack()
        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

        # Mediapipe Hands and KeyPointClassifier Initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.keypoint_classifier = KeyPointClassifier()
        self.keypoint_classifier_labels = ["paper", "rock", "scissors", "unknown"]

        pygame.mixer.music.play(loops=0, start=0.4)
        self.update_video()
        self.updateComputerChoice()
        self.root.mainloop()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB (OpenCV uses BGR)
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe Hands
            results = self.hands.process(frame_rgb)
            gesture_text = "No Hand Detected"
            # print(results.multi_hand_landmarks[0].landmark[0].x, results.multi_hand_landmarks[0].landmark[0].y, results.multi_hand_landmarks[0].landmark[0].z)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmark coordinates
                    landmark_list = self.calc_landmark_list(frame, hand_landmarks)

                    # Preprocess the landmarks
                    preprocessed_landmarks = self.pre_process_landmark(landmark_list)

                    # Classify the hand gesture
                    hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
                    gesture_text = self.keypoint_classifier_labels[hand_sign_id]

            # Display the classification result
            self.label.config(text=f"Gesture: {gesture_text}")

            # Convert the frame to a PIL Image
            img = Image.fromarray(frame_rgb)
            # Convert the PIL Image to an ImageTk object
            imgtk = ImageTk.PhotoImage(image=img)
            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        self.root.after(10, self.update_video)  # Update the video every 10ms

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
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

    def pointsToWin(self):
        pygame.mixer.music.play(loops=0, start=0.4)
        print(self.points.get())
        self.videoFrame.pack()
        self.pointsFrame.pack(padx=20, pady=20, fill='x')
        self.getNumpointsFrame.pack_forget()
        self.label.pack(pady=10)
        self.countdown_sound.play(loops=0)

    def updateComputerChoice(self):
        self.computer_choice = random.choice([0, 1, 2])  # Randomly choose between 0 (paper), 1 (rock), and 2 (scissors)
        if self.computer_choice == 1:
            image_path = "rock.jpg"
        elif self.computer_choice == 0:
            image_path = "paper.jpg"
        elif self.computer_choice == 2:
            image_path = "scissor.jpg"
        else:
            image_path = "thinking.jpg"
        static_image = Image.open(image_path).resize((620, 480), Resampling.LANCZOS)
        static_image_tk = ImageTk.PhotoImage(static_image)
        self.computerChoiceImg.config(image=static_image_tk)
        self.computerChoiceImg.image = static_image_tk  
    def updateScore(self):
        human_choice = self.keypoint_classifier_labels.index(self.label.cget("text").split(": ")[1])
        print( "Human choice: " , human_choice, "|  Computer choice: ", self.computer_choice)
        if human_choice == self.computer_choice:  # Tie   Paper=0 Rock=1 Scissor=2
            pass
        elif (human_choice == 0 and self.computer_choice == 2) or (human_choice == 1 and self.computer_choice == 0) or (human_choice == 2 and self.computer_choice == 1):
            self.computerPoints.set(self.computerPoints.get() + 1)
        else:
            self.humanPoints.set(self.humanPoints.get() + 1)
        self.humanPointsLabel.config(text=f"Your Points: {self.humanPoints.get()}")
        self.computerPointsLabel.config(text=f"Computer Points: {self.computerPoints.get()}")
        if self.humanPoints.get() >= self.points.get():
            self.label.config(text="You Win!")
            self.resetGame()
        elif self.computerPoints.get() >= self.points.get():
            self.label.config(text="Computer Wins!")
            self.resetGame()
        else:
            self.label.config(text="Choose an option")
        self.updateComputerChoice()
    def resetGame(self):
        self.humanPoints.set(0)
        self.computerPoints.set(0)
        self.humanPointsLabel.config(text=f"Your Points: {self.humanPoints.get()}")
        self.computerPointsLabel.config(text=f"Computer Points: {self.computerPoints.get()}")
        self.videoFrame.pack_forget()
        self.pointsFrame.pack_forget()
        self.getNumpointsFrame.pack(pady=20)
        self.label.pack_forget()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    GUI()
    print("Thanks for playing!")



