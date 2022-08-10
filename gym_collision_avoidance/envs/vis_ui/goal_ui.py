import gym
import pygame
from pygame.locals import *
import os
import cv2
import numpy as np

from gym_collision_avoidance.envs.collision_avoidance_env import CollisionAvoidanceEnv


def get_opencv_img_res(opencv_image):
    height, width = opencv_image.shape[:2]
    return width, height


def convert_opencv_img_to_pygame(opencv_image):
    """
    Convert OpenCV images for Pygame.
        see https://gist.github.com/radames/1e7c794842755683162b
    """
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
    # Generate a Surface for drawing images with Pygame based on OpenCV images
    pygame_image = pygame.surfarray.make_surface(rgb_image)

    return pygame_image


if __name__ == "__main__":

    # Initialize environment
    env = CollisionAvoidanceEnv()
    save_path = os.path.join(
        os.getcwd(), "gym_collision_avoidance/experiments/results_ui/"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    env.set_use_expert_action(1, True, "ig_greedy", False, 0.0, True)
    dummy_action = (
        np.zeros((1, 2))
        if isinstance(env.action_space, gym.spaces.Box)
        else np.zeros((1, 1), dtype=np.uint64)
    )

    # Get initial rendering
    obs = env.reset()
    opencv_image = env.render(mode="rgb_array")

    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()

    width, height = get_opencv_img_res(opencv_image)
    screen = pygame.display.set_mode((width, height))

    # Init variables
    image_clicked = False
    coord = (0, 0)
    agent_stopped = False

    running = True
    while running:
        # Convert OpenCV images for Pygame
        pygame_image = convert_opencv_img_to_pygame(opencv_image)
        # Draw image
        screen.blit(pygame_image, (0, 0))
        pygame.display.update()

        # Collect pygame events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # elif event.key == K_SPACE:
                #     if not agent_stopped:
                #         env.set_use_expert_action(
                #             1, False, "ig_greedy", False, 0.0, False
                #         )
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Set the x, y postions of the mouse click
                coord = event.pos
                image_clicked = True
                # if pygame_image.get_rect().collidepoint(*coord):
                # print("clicked on image at " + str(coord))

        # Step and render environment
        if image_clicked:
            env.set_new_human_goal(coord=coord)
            image_clicked = False
        obs, reward, game_over, info = env.step(dummy_action)
        opencv_image = env.render(mode="rgb_array")

        clock.tick(2)

    pygame.quit()
