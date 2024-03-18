# Create Objects
objects_list = []
vao_list = []

# Object 1
vertices1, indices1, indices_count1 = create_sphere_vertices(SURFACE_ROWS, SURFACE_COLS)
object1 = ObjectManager(vertices1, indices1, GRID_ROWS, GRID_COLS, GRID_SPACING, [1.0, 0.0, 0.0, 1.0])
object1.create_buffers()
vao1 = object1.vao
vao_list.append(vao1)
objects_list.append(object1)

# Object 2
vertices2, indices2, indices_count2 = create_sphere_vertices(SURFACE_ROWS, SURFACE_COLS)
object2 = ObjectManager(vertices2, indices2, GRID_ROWS, GRID_COLS, GRID_SPACING, [0.0, 1.0, 0.0, 1.0])
object2.create_buffers()
vao2 = object2.vao
vao_list.append(vao2)
objects_list.append(object2)

# Set up uniforms time projection and view
time_location = glGetUniformLocation(shpere_program, "time")
proj_location = glGetUniformLocation(shpere_program, "projection")
view_location = glGetUniformLocation(shpere_program, "view")
glUniformMatrix4fv(proj_location, 1, GL_FALSE, projection)
glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

running = True
frame_count = 0
zero_time = glfw.get_time()

while running:
    start_time = glfw.get_time()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    move_camera()
    view = cam.get_view_matrix()

    glUniform1f(time_location, start_time)
    glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

    # Render objects
    for i in range(len(objects_list)):
        glBindVertexArray(vao_list[i])
        glDrawElementsInstanced(GL_TRIANGLES, objects_list[i].indices_count, GL_UNSIGNED_INT, None, instances_area)

    # Reset frame
    glfw.swap_buffers(gwindow)
    frame_count, zero_time, fps = handle_events(frame_count, zero_time, start_time, gwindow)
    glfw.poll_events()

    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        [shader.delete() for shader in shaders_list]
        [object.delete_buffers() for object in objects_list]
        glfw.terminate()
        running = False

    frame_count += 1