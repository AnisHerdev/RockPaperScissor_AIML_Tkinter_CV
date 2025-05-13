import tkinter as tk
import itertools
import csv
import random
import pygame
import numpy as np
import cv2
from PIL import Image, ImageTk
from PIL.Image import Resampling
import mediapipe as mp
from model import KeyPointClassifier

try:
    pygame.mixer.init()
except pygame.error:
    print("Audio device not available. Disabling audio.")
    pygame.mixer = None

class GUI:
    def __init__(self):
        pygame.mixer.init()

        self.countdown_sound = pygame.mixer.Sound("countdown.mp3")
        pygame.mixer.music.load("game-start.mp3")
        self.root = tk.Tk()
        self.root.title("Rock Paper Scissors") # name of the window
        self.root.geometry("960x600")
        self.root.config(bg="light green")
        self.main_label = tk.Label(self.root, text="Rock Paper Scissors", font=("Arial", 24))
        self.main_label.pack(pady=20)

        self.points = tk.IntVar()
        self.points.set(3) # default value for points to win

        self.get_num_points_frame = tk.Frame(self.root)
        self.get_num_points_frame.config(bg="light green")
        self.get_num_points_frame.columnconfigure(0, weight=1)
        self.get_num_points_frame.columnconfigure(1, weight=1)
        self.get_num_points_frame.columnconfigure(2, weight=1)

        self.points_label = tk.Label(self.get_num_points_frame, text="Points to Win", font=('Arial', 16), bg="light green")
        self.points_label.grid(row=0, column=0, padx=10)
        self.points_entry = tk.Entry(self.get_num_points_frame, textvariable=self.points, font=('Arial', 16), width=5)
        self.points_entry.grid(row=0, column=1, padx=10)
        self.points_to_win_btn = tk.Button(self.get_num_points_frame, text="Start Game", font=('Arial', 16), command=self.points_to_win)
        self.points_to_win_btn.grid(row=0, column=2, padx=10)

        self.get_num_points_frame.pack(pady=20)

        self.points_frame = tk.Frame(self.root)
        self.points_frame.config(bg="light green")
        self.points_frame.columnconfigure(0, weight=1)
        self.points_frame.columnconfigure(1, weight=1)

        self.human_points = tk.IntVar()
        self.human_points.set(0)
        self.computer_points = tk.IntVar()
        self.computer_points.set(0)
        self.human_points_label = tk.Label(self.points_frame, text=f"Your Points: {self.human_points.get()}", font=('Arial', 16), bg="light green")
        self.human_points_label.grid(row=0, column=0, padx=10)
        self.computer_points_label = tk.Label(self.points_frame, text=f"Computer Points: {self.computer_points.get()}", font=('Arial', 16), bg="light green")
        self.computer_points_label.grid(row=0, column=1, padx=10)

        self.label = tk.Label(self.root, text="Choose an option", font=("Arial", 16))

        # Video Frame
        self.video_frame = tk.Frame(self.root)
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.columnconfigure(1, weight=1)
        self.canvas = tk.Canvas(self.video_frame, width=620, height=480)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.computer_choice_img = tk.Label(self.video_frame)
        self.computer_choice_img.grid(row=0, column=1, padx=10, pady=10)

        static_image = Image.open("scissor.jpg").resize((620, 480), Image.Resampling.LANCZOS)
        static_image_tk = ImageTk.PhotoImage(static_image)
        self.computer_choice_img.config(image=static_image_tk)
        self.computer_choice_img.image = static_image_tk  # Keep a reference to avoid garbage collection
        # self.btn = tk.Button(self.root, text="Computer", font=('Arial', 16), command=self.update_computer_choice) # UpdatecomputerChoice manually
        # self.btn.pack()
        # self.btn2 = tk.Button(self.root, text="ScoreUpdate", font=('Arial', 16), command=self.update_score) # UpdateScore manually
        # self.btn2.pack()
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

        self.winner_label = tk.Label(self.root, text="You Win!", font=("Arial", 16), bg="light green")
        pygame.mixer.music.play(loops=0, start=0.4)
        self.actions = ["paper", "rock", "scissors"]
        try:
            self.q_table = np.load("q_table.npy")
            print("Q-table loaded successfully.")
        except FileNotFoundError:
            self.q_table = np.zeros((3, 3))
            print("No Q-table found. Initialized a new Q-table.")
        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.update_video()
        self.is_running = True
        self.computer_choice = self.choose_action(random.randint(0, 2))  # Use Q-table to decide initial choice
        print("Computer choice: ", self.computer_choice)
        self.update_computer_choice()
        self.restart = tk.Button(self.root, text="Reset Game", font=('Arial', 16), command=self.reset_game)
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
            # print("$",results.multi_hand_landmarks[0].landmark[0].x, results.multi_hand_landmarks[0].landmark[0].y, results.multi_hand_landmarks[0].landmark[0].z)
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

    def points_to_win(self):
        pygame.mixer.music.play(loops=0, start=0.4)
        print(self.points.get())
        self.video_frame.pack()
        self.points_frame.pack(padx=20, pady=20, fill='x')
        self.get_num_points_frame.pack_forget()
        self.winner_label.pack_forget()
        self.label.pack(pady=10)
        self.is_running = True
        self.restart.pack(pady=20)
        self.start_game()
    
    def start_game(self):
        if not self.is_running:
            return
        self.countdown_sound.stop()
        self.countdown_sound.play(loops=0)
        static_image = Image.open("thinking.jpg").resize((620, 480), Resampling.LANCZOS)
        static_image_tk = ImageTk.PhotoImage(static_image)
        self.computer_choice_img.config(image=static_image_tk)
        self.computer_choice_img.image = static_image_tk  
        self.update_score_timer = self.root.after(3400, self.update_score)
        self.start_game_timer = self.root.after(6000, self.start_game)

    def update_computer_choice(self):
        if self.computer_choice == 1:
            print("Computer choice: Rock", self.computer_choice)
            image_path = "rock.jpg"
        elif self.computer_choice == 0:
            print("Computer choice: Paper", self.computer_choice)
            image_path = "paper.jpg"
        elif self.computer_choice == 2: 
            print("Computer choice: Scissors", self.computer_choice)
            image_path = "scissor.jpg"
        else:
            print("Computer choice: Unknown", self.computer_choice)
            image_path = "thinking.jpg"
        static_image = Image.open(image_path).resize((620, 480), Resampling.LANCZOS)
        static_image_tk = ImageTk.PhotoImage(static_image)
        self.computer_choice_img.config(image=static_image_tk)
        self.computer_choice_img.image = static_image_tk  

    def update_score(self):
        if not self.is_running:
            return
        pygame.mixer.music.play(loops=0, start=0.4)
        try:
            human_choice = self.keypoint_classifier_labels.index(self.label.cget("text").split(": ")[1])
        except ValueError:
            return

        self.computer_choice = self.choose_action(human_choice)        
        # Q-learning update
        # print("Computer choice:", self.computer_choice)
        self.update_computer_choice()
        print( "Human choice: " , human_choice, "|  Computer choice: ", self.computer_choice)
        if human_choice == self.computer_choice:  # Tie     Paper=0 Rock=1 Scissor=2
            reward = 0
        elif (human_choice == 0 and self.computer_choice == 2) or \
             (human_choice == 1 and self.computer_choice == 0) or \
             (human_choice == 2 and self.computer_choice == 1):
            self.computer_points.set(self.computer_points.get() + 1)
            reward = 1
        else:
            self.human_points.set (self.human_points.get() + 1)
            reward =-1
        with open('output.csv', mode='a',newline='') as file: 
            csv_writer = csv.writer(file)    
            csv_writer.writerow([human_choice, self.computer_choice, reward])

        self.human_points_label.config(text=f"Your Points: {self.human_points.get()}")
        self.computer_points_label.config(text=f"Computer Points: {self.computer_points.get()}")
        if self.human_points.get() >= self.points.get():
            self.winner_label.config(text="You Win!")
            self.winner_label.pack(pady=10)
            self.root.after(3000, self.reset_game)
            # self.reset_game()
        elif self.computer_points.get() >= self.points.get():
            self.winner_label.config(text="Computer Wins!")
            self.winner_label.pack(pady=10)
            self.root.after(3000, self.reset_game)
            # self.reset_game()
        else:
            self.label.config(text="Choose an option")
        self.q_table[human_choice, self.computer_choice] += self.alpha * (reward + self.gamma * np.max(self.q_table[self.computer_choice]) - self.q_table[human_choice, self.computer_choice])
        # print("Q-table:\n", self.q_table)


    def reset_game(self):
        self.countdown_sound.stop()
        self.is_running = False
        print("Resetting is_running to false...")
        # Cancel any scheduled calls to update_score or start_game
        if hasattr(self, 'update_score_timer'):
            self.root.after_cancel(self.update_score_timer)
        if hasattr(self, 'start_game_timer'):
            self.root.after_cancel(self.start_game_timer)
        self.human_points.set(0)
        self.computer_points.set(0)
        self.human_points_label.config(text=f"Your Points: {self.human_points.get()}")
        self.computer_points_label.config(text=f"Computer Points: {self.computer_points.get()}")
        self.video_frame.pack_forget()
        self.points_frame.pack_forget()
        self.get_num_points_frame.pack(pady=20)
        self.label.pack_forget()
        self.restart.pack_forget()

    def choose_action(self, state):
        """self.Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)  # Explore
        return np.argmax(self.q_table[state])  # Exploit
    
    # def __del__(self):
    #     if hasattr(self, 'cap') and self.cap.isOpened():
    #         self.cap.release()
    #     np.save("q_table.npy", self.q_table)
    #     print("Q-table saved successfully.")

if __name__ == "__main__":
    GUI()
    print("Thanks for playing!")