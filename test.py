
def create_tunnel_vertices(self):
    vertices = []
    indices = []
    segment_angle = 2 * np.pi / self.grid_rows

    # Generate vertices for the tunnel
    for z in range(2):  # Two layers: start and end of the tunnel segment
        for i in range(self.grid_rows):
            angle = i * segment_angle
            x = cos(angle)
            y = sin(angle)
            vertices.extend([x, y, z * self.grid_cols])  # Place vertex at start or end

    # Generate indices for the tunnel's quads
    for i in range(self.grid_rows):
        # Calculate current and next segment indices (wrapping around)
        current_segment = i
        next_segment = (i + 1) % self.grid_rows

        # Bottom layer to top layer indices
        bottom_current = current_segment
        bottom_next = next_segment
        top_current = current_segment + self.grid_rows
        top_next = next_segment + self.grid_rows

        # Create two triangles for each quad
        indices.extend([bottom_current, top_current, bottom_next])
        indices.extend([bottom_next, top_current, top_next])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)