from pyrr import Vector3, Quaternion, Matrix44, vector, quaternion
from math import radians

class Camera:
    def __init__(self):
        self.camera_pos = Vector3([0.0, 10.0, -30.0])
        self.orientation = Quaternion([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        self.mouse_sense = 0.25
        self.camera_front = Vector3([0.0, 0.0, 0.0])
        self.camera_up = Vector3([0.0, 1.0, 0.0])
        self.camera_right = Vector3([1.0, 0.0, 0.0])
        self.forward = vector.normalize(quaternion.apply_to_vector(self.orientation, Vector3([0.0, 0.0, -1.0])))
        self.side = vector.normalize(quaternion.apply_to_vector(self.orientation, Vector3([1.0, 0.0, 0.0])))
        self.dive = Vector3([0.0, 1.0, 0.0])

    def get_view_matrix(self):
        rotation_matrix = Matrix44.from_quaternion(self.orientation)
        translation_matrix = Matrix44.from_translation(-self.camera_pos)
        view_matrix = rotation_matrix * translation_matrix
        return view_matrix.inverse

    def process_mouse_movements(self, x_offset, y_offset):
        x_offset *= self.mouse_sense
        y_offset *= -self.mouse_sense
        # Creating quaternions for the pitch (x rotation) and yaw (y rotation)
        pitch_quat = quaternion.create_from_x_rotation(radians(y_offset))
        yaw_quat = quaternion.create_from_y_rotation(radians(x_offset))
        # Update the orientation quaternion
        self.orientation = quaternion.cross(self.orientation, pitch_quat)
        self.orientation = quaternion.cross(yaw_quat, self.orientation)
        self.orientation = quaternion.normalize(self.orientation)

    def process_keyboard(self, direction, velocity):

        #self.forward = vector.normalize(quaternion.apply_to_vector(self.orientation, Vector3([0.0, 0.0, -1.0])))
        #self.side = vector.normalize(quaternion.apply_to_vector(self.orientation, Vector3([1.0, 0.0, 0.0])))

        if direction == "FORWARD":
            self.camera_pos -= self.forward * velocity
        if direction == "BACKWARD":
            self.camera_pos += self.forward * velocity
        if direction == "LEFT":
            self.camera_pos += self.side * velocity
        if direction == "RIGHT":
            self.camera_pos -= self.side * velocity
        if direction == "UP":
            self.camera_pos += self.dive * velocity
        if direction == "DOWN":
            self.camera_pos -= self.dive * velocity
