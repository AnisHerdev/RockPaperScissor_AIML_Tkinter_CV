import tkinter as tk
from app import main  
import cv2
from PIL import Image, ImageTk

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rock Paper Scissors") # name of the window
        self.root.geometry("800x500")
        self.root.config(bg="light green")
        self.mainLabel = tk.Label(self.root, text="Rock Paper Scissors", font=("Arial", 24))
        self.mainLabel.pack(pady=20)

        self.points = tk.IntVar()
        self.points.set(3) # default value for points to win

        self.pointsFrame = tk.Frame(self.root)
        self.pointsFrame.config(bg="light green")
        self.pointsFrame.columnconfigure(0,weight=1)
        self.pointsFrame.columnconfigure(1,weight=1)
        self.pointsFrame.columnconfigure(2,weight=1)

        self.pointsLabel = tk.Label(self.pointsFrame, text="Points to Win", font=('Arial',16),bg="light green")
        self.pointsLabel.grid(row = 0,column=0, padx=10)
        self.pointsEntry = tk.Entry(self.pointsFrame, textvariable=self.points, font=('Arial',16),width=5)
        self.pointsEntry.grid(row = 0,column=1, padx=10)
        self.pointsToWinBtn = tk.Button(self.pointsFrame, text="Start Game", font=('Arial',16),command=self.pointsToWin)
        self.pointsToWinBtn.grid(row = 0,column=2,padx=10)

        self.pointsFrame.pack(pady=20)

        self.buttonFrame = tk.Frame(self.root)
        self.buttonFrame.columnconfigure(0,weight=1)
        self.buttonFrame.columnconfigure(1,weight=1)
        self.buttonFrame.columnconfigure(2,weight=1)

        self.rockBtn = tk.Button(self.buttonFrame, text="Rock" , font=('Arial',16),command=self.rockChosen)
        self.rockBtn.grid(row=0,column=0,sticky=tk.W+tk.E)
        self.paperBtn = tk.Button(self.buttonFrame, text="Paper" , font=('Arial',16),command=self.paperChosen)
        self.paperBtn.grid(row=0,column=1,sticky=tk.W+tk.E)
        self.scissorBtn = tk.Button(self.buttonFrame, text="Scissor" , font=('Arial',16),command=self.scissorChosen)
        self.scissorBtn.grid(row=0,column=2,sticky=tk.W+tk.E)

        self.label = tk.Label(self.root, text="Choose an option", font=("Arial", 16))

        # Video Frame
        self.videoFrame = tk.Frame(self.root)
        self.videoFrame.pack(pady=10)
        self.canvas = tk.Canvas(self.videoFrame, width=640, height=480)
        self.canvas.pack()

        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path
        self.update_video()

        self.root.mainloop()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB (OpenCV uses BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL Image
            img = Image.fromarray(frame)
            # Convert the PIL Image to an ImageTk object
            imgtk = ImageTk.PhotoImage(image=img)
            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        self.root.after(10, self.update_video)  # Update the video every 10ms

    def scissorChosen(self):
        self.label.config(text="You chose Scissor")
        # self.label.update_idletasks()    
    def rockChosen(self):
        self.label.config(text="You chose Rock")
    def paperChosen(self):
        self.label.config(text="You chose Paper")
    def pointsToWin(self):
        print(self.points.get())
        self.pointsFrame.destroy()
        self.buttonFrame.pack(padx=20, pady=20, fill='x')
        self.label.pack(pady=10)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    GUI()
    print("Thanks for playing!")



