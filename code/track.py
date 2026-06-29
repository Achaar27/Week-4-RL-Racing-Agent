import math
import pygame

# Screen size
SCREEN_W = 800
SCREEN_H = 600

# Centre of the track
CX = 400
CY = 300

# Outer ellipse size
OUTER_A = 320
OUTER_B = 240

# Inner ellipse size
INNER_A = 180
INNER_B = 100


def on_track(px, py):
    # Move point to be relative to centre
    dx = px - CX
    dy = py - CY

    # Check ellipse equation for both ellipses
    outer = (dx / OUTER_A) ** 2 + (dy / OUTER_B) ** 2
    inner = (dx / INNER_A) ** 2 + (dy / INNER_B) ** 2

    # On track means inside outer but outside inner
    return outer <= 1.0 and inner >= 1.0


def track_progress(px, py):
    # Get angle of car relative to track centre
    dx = px - CX
    dy = py - CY
    angle = math.atan2(dy, dx)

    # Convert angle to a value between 0 and 1
    return (angle % (2 * math.pi)) / (2 * math.pi)


# Run visual test when this file is run directly
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

        # Draw the track pixel by pixel
        for py in range(0, SCREEN_H, 4):
            for px in range(0, SCREEN_W, 4):
                if on_track(px, py):
                    pygame.draw.rect(screen, (75, 75, 75), (px, py, 4, 4))

        # Draw start line in red
        pygame.draw.line(screen, (255, 50, 50),
                         (CX + INNER_A, CY),
                         (CX + OUTER_A, CY), 3)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()