import math
import pygame


SCREEN_W = 800
SCREEN_H = 600
CX = 400
CY = 300
OUTER_A = 320
OUTER_B = 240
INNER_A = 180
INNER_B = 100


def on_track(px, py):
    dx = px - CX
    dy = py - CY
    outer = (dx / OUTER_A) ** 2 + (dy / OUTER_B) ** 2
    inner = (dx / INNER_A) ** 2 + (dy / INNER_B) ** 2
    return outer <= 1.0 and inner >= 1.0


def track_progress(px, py):
    dx = px - CX
    dy = py - CY
    angle = math.atan2(dy, dx)
    return (angle % (2 * math.pi)) / (2 * math.pi)



if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Track Test")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((20, 20, 20))

       
        for py in range(0, SCREEN_H, 4):
            for px in range(0, SCREEN_W, 4):
                if on_track(px, py):
                    pygame.draw.rect(screen, (75, 75, 75), (px, py, 4, 4))

      
        pygame.draw.line(screen, (255, 50, 50),
                         (CX + INNER_A, CY),
                         (CX + OUTER_A, CY), 3)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
