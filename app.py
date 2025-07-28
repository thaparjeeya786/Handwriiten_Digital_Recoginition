import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Constants
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
PRIDICT = True

# Load model
MODEL = load_model("unit_model.h5")

LABELS = {
    0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR",
    5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE"
}

# Initialize Pygame
pygame.init()
FONT = pygame.font.SysFont("Arial", 24)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

# Tracking states
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
                rect_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
                rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
                rect_max_y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)

                # Extract image from pygame surface
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
                number_xcord = []
                number_ycord = []

                if IMAGESAVE:
                    cv2.imwrite(f"image{image_cnt}.png", img_arr)
                    image_cnt += 1

                if PRIDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0

                    prediction = MODEL.predict(image.reshape(1, 28, 28, 1))
                    label = str(LABELS[np.argmax(prediction)])

                    # Draw rectangle
                    pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

                    # Show predicted label below rectangle
                    textSurface = FONT.render(label, True, RED)
                    textRect = textSurface.get_rect()
                    textRect.center = ((rect_min_x + rect_max_x) // 2, rect_max_y + 20)

                    DISPLAYSURF.blit(textSurface, textRect)

        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
