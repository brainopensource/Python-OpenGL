from pyrr import Vector3, vector, vector3, matrix44
from math import sin, cos, radians


class Camera:
    def __init__(self):
        self.camera_pos = Vector3([0.0, 1.0, 0.0])
        self.camera_front = Vector3([0.0, 0.0, -1.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        self.camera_right = Vector3([1.0, 0.0, 0.0])
        self.dive = Vector3([0.0, 1.0, 0.0])
        self.mouse_sense = 0.1
        self.yaw = -90
        self.pitch = -30


    def get_view_matrix(self):
        return matrix44.create_look_at(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)


    def process_mouse_movements(self, x, y):
        x *= self.mouse_sense
        y *= self.mouse_sense
        self.yaw += x
        self.pitch += y
        #if lock_pitch:
        #    self.pitch = max(min(self.pitch, 90), -90)
        self.update_camera_vectors()


    def update_camera_vectors(self):
        front = Vector3([0.0, 0.0, 0.0])
        front.x = cos(radians(self.yaw)) * cos(radians(self.pitch))
        front.y = sin(radians(self.pitch))
        front.z = sin(radians(self.yaw)) * cos(radians(self.pitch))
        self.camera_front = vector.normalize(front)
        self.camera_right = vector.normalize(vector3.cross(self.camera_front, Vector3([0.0, 1.0, 0.0])))
        self.camera_up = vector.normalize(vector3.cross(self.camera_right, self.camera_front))


    def process_keyboard(self, direction, velocity):
        if direction == "FORWARD":
            self.camera_pos += self.camera_front * velocity
        if direction == "BACKWARD":
            self.camera_pos -= self.camera_front * velocity
        if direction == "LEFT":
            self.camera_pos -= self.camera_right * velocity
        if direction == "RIGHT":
            self.camera_pos += self.camera_right * velocity
        if direction == "UP":
            self.camera_pos += self.dive * velocity
        if direction == "DOWN":
            self.camera_pos -= self.dive * velocity

