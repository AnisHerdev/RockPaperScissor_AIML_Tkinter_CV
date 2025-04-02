# from playsound import playsound # Import the playsound function from the library.

# filename = 'countdown.mp3'  
# playsound(filename) # Use playsound to play the audio file.


import pygame

filename = 'countdown.mp3'
pygame.mixer.init()
pygame.mixer.music.load(filename)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pass