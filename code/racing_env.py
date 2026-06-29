import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from track import on_track, track_progress, CX, CY, INNER_A, OUTER_A, SCREEN_W, SCREEN_H

RAY_MAX = 200.0

class RacingEnv(gym.Env):

    def __init__(self, render_mode=None, progress_scale=10.0, time_penalty=-0.01, max_speed=5.0, accel=0.5):
        super().__init__()

        self.render_mode = render_mode
        self._screen = None
        self._clock = None

        # 8 inputs - speed, sin and cos of heading, and 5 rays
        low  = np.array([0, -1, -1, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1,  1,  1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 5 actions the car can take
        self.action_space = spaces.Discrete(5)

        # car physics
        self.max_speed = max_speed
        self.accel = accel
        self.brake_force = 0.8
        self.drag = 0.05
        self.steer = 3.5

        # rays shoot out in these directions from the car
        self.ray_angles = [-90, -45, 0, 45, 90]

        # car position and state
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0

        self._prev_progress = 0.0
        self._lap_crossings = 0
        self._step_count = 0

        # reward settings - can change these for experiments
        self.progress_scale = progress_scale
        self.time_pen = time_penalty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # start on the right side of the oval facing up
        self.x = float(CX + OUTER_A - (OUTER_A - INNER_A) // 2)
        self.y = float(CY)
        self.heading = -math.pi / 2
        self.speed = 0.5

        self._prev_progress = track_progress(self.x, self.y)
        self._lap_crossings = 0
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # apply whatever action the agent chose
        if action == 1:
            self.speed = min(self.max_speed, self.speed + self.accel)
        elif action == 2:
            self.speed = max(0.0, self.speed - self.brake_force)
        elif action == 3:
            self.heading += math.radians(self.steer)
        elif action == 4:
            self.heading -= math.radians(self.steer)
        # action 0 means do nothing

        # slow down a bit every step due to drag
        self.speed = max(0.0, self.speed - self.drag)

        # move the car forward
        self.x += math.cos(self.heading) * self.speed
        self.y += math.sin(self.heading) * self.speed

        # did the car go off track?
        crashed = not on_track(self.x, self.y)

        # how much further around the track did the car move
        current_progress = track_progress(self.x, self.y)
        delta = current_progress - self._prev_progress

        # handle the lap wraparound (progress goes from 0.99 back to 0.0)
        if delta < -0.5:
            delta += 1.0
            self._lap_crossings += 1
        elif delta > 0.5:
            delta -= 1.0

        self._prev_progress = current_progress

        # calculate reward
        progress_reward = delta * self.progress_scale
        lap_bonus = 5.0 if self._lap_crossings > 0 else 0.0
        self._lap_crossings = 0

        if crashed:
            reward = -3.0
        else:
            reward = progress_reward + self.time_pen + lap_bonus

        terminated = crashed
        truncated = self._step_count >= 1000

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        speed_norm = self.speed / self.max_speed
        sin_h = math.sin(self.heading)
        cos_h = math.cos(self.heading)
        rays = self._cast_rays()
        return np.array([speed_norm, sin_h, cos_h] + rays, dtype=np.float32)

    def _cast_rays(self):
        # shoot rays in 5 directions and measure how far until hitting a wall
        distances = []
        for offset in self.ray_angles:
            angle = self.heading + math.radians(offset)
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist = 0.0

            while dist < RAY_MAX:
                dist += 3.0
                if not on_track(self.x + dx * dist, self.y + dy * dist):
                    break

            distances.append(min(dist, RAY_MAX) / RAY_MAX)
        return distances

    def _render_frame(self):
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("RL Oval Racer")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self._screen.fill((20, 20, 20))

        # draw the track
        for py in range(0, SCREEN_H, 4):
            for px in range(0, SCREEN_W, 4):
                if on_track(px, py):
                    pygame.draw.rect(self._screen, (75, 75, 75), (px, py, 4, 4))

        # start line
        pygame.draw.line(self._screen, (255, 50, 50),
                         (CX + INNER_A, CY),
                         (CX + OUTER_A, CY), 3)

        # draw rays
        cx, cy = int(self.x), int(self.y)
        for offset in self.ray_angles:
            angle = self.heading + math.radians(offset)
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist = 0.0
            while dist < RAY_MAX:
                dist += 3.0
                if not on_track(self.x + dx * dist, self.y + dy * dist):
                    break
            ex = int(cx + dx * dist)
            ey = int(cy + dy * dist)
            pygame.draw.line(self._screen, (0, 170, 230), (cx, cy), (ex, ey), 1)

        # draw car
        pygame.draw.circle(self._screen, (255, 60, 60), (cx, cy), 10)

        # show some info on screen
        font = pygame.font.SysFont(None, 28)
        self._screen.blit(font.render(f"Speed   : {self.speed:.2f}", True, (255, 255, 255)), (10, 10))
        self._screen.blit(font.render(f"Step    : {self._step_count}", True, (255, 255, 255)), (10, 35))
        self._screen.blit(font.render(f"Progress: {track_progress(self.x, self.y):.2f}", True, (255, 255, 255)), (10, 60))

        pygame.display.flip()
        self._clock.tick(60)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None