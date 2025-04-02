# from playsound import playsound # Import the playsound function from the library.

# filename = 'countdown.mp3'  
# playsound(filename) # Use playsound to play the audio file.


import pygame

filename = 'countdown.mp3'
pygame.mixer.init()
pygame.mixer.music.load(filename)
pygame.mixer.music.play(loops=5,start=1)

pause = input("Press Enter to stop the music...")
pygame.mixer.music.pause()
play = input("Press Enter to play the music again...")

pygame.mixer.music.fadeout(5000)
while pygame.mixer.music.get_busy():
    pass