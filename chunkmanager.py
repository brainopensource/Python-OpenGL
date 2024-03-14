

class ChunkManager:
    def __init__(self, render_distance):
        self.render_distance = render_distance
        self.loaded_chunks = {}  # Key: (chunkX, chunkZ), Value: Chunk data
        self.chunk_size = 50  # Assuming each chunk is 50x50 units

    def get_chunk_index(self, position):
        # Calculate chunk indices based on world position and chunk size
        chunkX = int(position[0] // self.chunk_size)
        chunkZ = int(position[1] // self.chunk_size)
        return (chunkX, chunkZ)

    def update_chunks(self, player_position):
        # Determine which chunks need to be loaded based on player position
        current_index = self.get_chunk_index(player_position)
        chunks_to_load = self.get_surrounding_chunks(current_index)

        # Load necessary chunks
        for chunk_index in chunks_to_load:
            if chunk_index not in self.loaded_chunks:
                self.load_chunk(chunk_index)

        # Unload distant chunks
        distant_chunks = [index for index in self.loaded_chunks if index not in chunks_to_load]
        for index in distant_chunks:
            self.unload_chunk(index)

    def get_surrounding_chunks(self, current_index):
        # Determine indices of surrounding chunks to load
        surrounding_indices = []
        for i in range(-self.render_distance, self.render_distance + 1):
            for j in range(-self.render_distance, self.render_distance + 1):
                index = (current_index[0] + i, current_index[1] + j)
                surrounding_indices.append(index)
        return surrounding_indices

    def load_chunk(self, chunk_index):
        # Generate or retrieve chunk data and add it to loaded_chunks
        # Placeholder for chunk generation or loading logic
        chunk_data = "Generated or loaded chunk data"
        self.loaded_chunks[chunk_index] = chunk_data

    def unload_chunk(self, chunk_index):
        # Unload chunk data, potentially saving state if necessary
        del self.loaded_chunks[chunk_index]

