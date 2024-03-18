import random
import threading
import numpy as np
import pygame
import pygame.midi
import time as t


class OptimizedPendulum:
    # Class-level constants for improved readability and performance
    DECAY_FACTOR = 0.9
    ANGLE_THRESHOLD = np.radians(0.1)  # Threshold for stopping the pendulum
    VELOCITY = 64  # MIDI velocity
    CHANNEL = 0  # Default MIDI channel
    T_WAIT_MIDI = 20

    def __init__(self, origin, length, ball_r, period, max_angle_rad, sound_angle_rad, midi_out, instrument, xnote):
        self.origin = origin
        self.length = round(length)
        self.ball_r = ball_r
        self.period = period
        self.max_angle = max_angle_rad
        self.sound_angle = sound_angle_rad
        self.angle = np.radians(90)  # Convert starting angle to radians
        self.prev_angle = 0  # Previous angle for oscillation check
        self.midi_out = midi_out
        self.instrument = instrument
        self.xnote = xnote
        self.stopped = False
        self.sound_played = False
        self.trail = []
        self.trail_included = True
        self.ball_pos = self.calculate_ball_pos(self.angle)
        # Set the instrument once to avoid repetitive calls
        self.midi_out.set_instrument(self.instrument, self.CHANNEL)

    def calculate_ball_pos(self, angle):
        """Calculate the ball's position based on the current angle."""
        return np.array([self.origin[0] + self.length * np.sin(angle),
                         self.origin[1] + self.length * np.cos(angle)])

    def play_sound(self):
        """Play sound without blocking the main thread."""
        self.midi_out.set_instrument(self.instrument, self.CHANNEL)
        self.midi_out.note_on(self.xnote, self.VELOCITY, self.CHANNEL)
        pygame.time.wait(self.T_WAIT_MIDI)
        self.midi_out.note_off(self.xnote, self.VELOCITY, self.CHANNEL)

    def play_sound_threaded(self):
        """Handle sound playback in a separate thread."""
        thread = threading.Thread(target=self.play_sound)
        thread.start()

    def update(self, time):
        """Update the pendulum's state."""
        if not self.stopped:
            angular_displacement = self.max_angle * np.sin((2 * np.pi / self.period) * time)
            self.angle = angular_displacement
            self.ball_pos = self.calculate_ball_pos(self.angle)

            # Generic scenario for angle check with variable threshold
            if self.angle * self.prev_angle <= 0:
                self.play_sound_threaded()
                self.max_angle *= self.DECAY_FACTOR
                self.sound_played = False

            if abs(self.max_angle) < self.ANGLE_THRESHOLD:
                self.stopped = True

            self.prev_angle = self.angle

    def draw(self, screen):
        """Draw the pendulum and its trail."""
        pygame.draw.circle(screen, self.get_color_based_on_angle(), (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_r)
        if self.trail_included:
            self.draw_trail(screen)

    def draw_trail(self, screen):
        """Draw the trail of the pendulum."""
        for i, (pos, color) in enumerate(self.trail):
            trail_surface = pygame.Surface((self.ball_r*2, self.ball_r*2), pygame.SRCALPHA)
            opacity = int(255 * (i + 1) / len(self.trail))
            pygame.draw.circle(trail_surface, (*color, opacity), (self.ball_r, self.ball_r), self.ball_r)
            screen.blit(trail_surface, (int(pos[0]) - self.ball_r, int(pos[1]) - self.ball_r))

    def get_color_based_on_angle(self):
        """Map the pendulum's angle to a color."""
        color_ratio = abs(self.angle) / abs(self.max_angle)
        return (int(255 * (1 - color_ratio)), int(255 * color_ratio), 0)


def time_interface(tela, start):
    # Interface
    tempo_decorrido = t.time() - start
    minutos = int(tempo_decorrido // 60)
    segundos = int(tempo_decorrido % 60)
    milissegundos = int((tempo_decorrido * 1000) % 1000)
    tempo_texto = f"{minutos:02d}:{segundos:02d}:{milissegundos:03d}"
    fonte = pygame.font.Font(None, 36)
    texto_renderizado = fonte.render(tempo_texto, True, (255, 255, 255))
    tela.blit(texto_renderizado, (10, 10))  # posição 10 10 default


def generate_custom_list_reset(xlen):
    # Define the start and end values of the MIDI scale
    start_value = 22
    end_value = 108

    # Generate the list
    midi_list = [(start_value + i) % (end_value - start_value + 1) + start_value for i in range(xlen)]

    return midi_list


def generate_custom_list_high(xlen):
    # Starting value
    start_value = 22
    # Maximum value allowed in the list
    max_value = 108

    # Generate the list
    # Use min() to ensure the value does not exceed max_value
    custom_list = [min(start_value + i, max_value) for i in range(xlen)]

    return custom_list


def generate_custom_list(xlen):
    start_value = 60
    end_value = 108
    direction = 1  # 1 for increasing, -1 for decreasing
    current_value = start_value
    midi_list = []

    for _ in range(xlen):
        midi_list.append(current_value)

        # Determine the next value based on direction
        next_value = current_value + direction

        # Check and switch direction if limits are reached
        if next_value > end_value:
            direction = -1
            next_value = end_value - 1  # Start decreasing from one step below the max
        elif next_value < start_value:
            direction = 1
            next_value = start_value + 1  # Start increasing from one step above the min

        # Update current value for the next iteration
        current_value = next_value

    return midi_list


def main():
    pygame.init()
    pygame.midi.init()
    midi_out = pygame.midi.Output(0)

    height = 1920
    width = 1080
    screen = pygame.display.set_mode((height, width), pygame.DOUBLEBUF)
    fps = 120
    num_pendulos = 100

    # Initialize multiple pendulum instances
    pendulums = []
    max_angle_rad = np.radians(-90)
    origin = (height / 2, width / 6)
    l_0 = 200
    delta_l = ((width / 1.5 - l_0) / (num_pendulos - 1)) * 2
    r_0 = 4
    delta_r = 200
    instrument = 0
    xnote_list = generate_custom_list(num_pendulos)
    print(xnote_list)


    for i in range(num_pendulos):
        ball_r = r_0 + i * (1 / delta_r)
        length = l_0 + i * delta_l
        period = 5 + i / 20
        sound_angle_rad = np.radians(-90)
        xnote = xnote_list[i]
        pendulums.append(OptimizedPendulum(origin, length, ball_r,
                                           period, max_angle_rad, sound_angle_rad,
                                           midi_out, instrument, xnote))

    running = True
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill((0, 0, 0))
        current_time = (pygame.time.get_ticks() - start_time) / 1000

        pygame.draw.line(screen, (255, 255, 255), origin, (height / 2, width * 0.9), 4)

        for pendulum in pendulums:
            pendulum.update(current_time)
            pendulum.draw(screen)

        # Draw interface
        time_interface(screen, start_time)

        pygame.display.flip()
        clock.tick(fps)

    midi_out.close()
    pygame.midi.quit()
    pygame.quit()

if __name__ == "__main__":
    main()