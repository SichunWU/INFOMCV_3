import time
import cv2
import glm
import glfw
import numpy as np
import p3
from engine.base.program import get_linked_program
from engine.renderable.model import Model
from engine.buffer.texture import *
from engine.buffer.hdrbuffer import HDRBuffer
from engine.buffer.blurbuffer import BlurBuffer
from engine.effect.bloom import Bloom
from assignment import draw_mesh, set_voxel_positions, generate_grid, get_cam_positions, get_cam_rotation_matrices
from engine.camera import Camera
from engine.config import config

cube, hdrbuffer, blurbuffer, lastPosX, lastPosY = None, None, None, None, None
firstTime = True
window_width, window_height = config['window_width'], config['window_height']
camera = Camera(glm.vec3(0, 100, 0), pitch=-90, yaw=0, speed=40)
pressNum = 0

trainedGMMs = []

def draw_objs(obj, program, perspective, light_pos, texture, normal, specular, depth):
    program.use()
    program.setMat4('viewProject', perspective * camera.get_view_matrix())
    program.setVec3('viewPos', camera.position)
    program.setVec3('light_pos', light_pos)

    glActiveTexture(GL_TEXTURE1)
    program.setInt('mat.diffuseMap', 1)
    texture.bind()

    glActiveTexture(GL_TEXTURE2)
    program.setInt('mat.normalMap', 2)
    normal.bind()

    glActiveTexture(GL_TEXTURE3)
    program.setInt('mat.specularMap', 3)
    specular.bind()

    glActiveTexture(GL_TEXTURE4)
    program.setInt('mat.depthMap', 4)
    depth.bind()
    program.setFloat('mat.shininess', 128)
    program.setFloat('mat.heightScale', 0.12)

    obj.draw_multiple(program)


def main():
    global hdrbuffer, blurbuffer, cube, window_width, window_height

    if not glfw.init():
        print('Failed to initialize GLFW.')
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.SAMPLES, config['sampling_level'])

    if config['fullscreen']:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        window_width, window_height = mode.size.window_width, mode.size.window_height
        window = glfw.create_window(mode.size.window_width,
                                    mode.size.window_height,
                                    config['app_name'],
                                    glfw.get_primary_monitor(),
                                    None)
    else:
        window = glfw.create_window(window_width, window_height, config['app_name'], None, None)
    if not window:
        print('Failed to create GLFW Window.')
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_framebuffer_size_callback(window, resize_callback)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    program = get_linked_program('resources/shaders/vert.vs', 'resources/shaders/frag.fs')
    depth_program = get_linked_program('resources/shaders/shadow_depth.vs', 'resources/shaders/shadow_depth.fs')
    blur_program = get_linked_program('resources/shaders/blur.vs', 'resources/shaders/blur.fs')
    hdr_program = get_linked_program('resources/shaders/hdr.vs', 'resources/shaders/hdr.fs')

    blur_program.use()
    blur_program.setInt('image', 0)

    hdr_program.use()
    hdr_program.setInt('sceneMap', 0)
    hdr_program.setInt('bloomMap', 1)

    window_width_px, window_height_px = glfw.get_framebuffer_size(window)

    hdrbuffer = HDRBuffer()
    hdrbuffer.create(window_width_px, window_height_px)
    blurbuffer = BlurBuffer()
    blurbuffer.create(window_width_px, window_height_px)

    bloom = Bloom(hdrbuffer, hdr_program, blurbuffer, blur_program)

    light_pos = glm.vec3(0.5, 0.5, 0.5)
    perspective = glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])

    cam_rot_matrices = get_cam_rotation_matrices()
    cam_shapes = [Model('resources/models/camera.json', cam_rot_matrices[c]) for c in range(4)]
    square = Model('resources/models/square.json')
    cube = Model('resources/models/cube.json')
    texture = load_texture_2d('resources/textures/diffuse.jpg')
    texture_grid = load_texture_2d('resources/textures/diffuse_grid.jpg')
    normal = load_texture_2d('resources/textures/normal.jpg')
    normal_grid = load_texture_2d('resources/textures/normal_grid.jpg')
    specular = load_texture_2d('resources/textures/specular.jpg')
    specular_grid = load_texture_2d('resources/textures/specular_grid.jpg')
    depth = load_texture_2d('resources/textures/depth.jpg')
    depth_grid = load_texture_2d('resources/textures/depth_grid.jpg')

    grid_positions, grid_colors = generate_grid(config['world_width'], config['world_width'])
    square.set_multiple_positions(grid_positions, grid_colors)

    cam_positions, cam_colors = get_cam_positions()
    for c, cam_pos in enumerate(cam_positions):
        cam_shapes[c].set_multiple_positions([cam_pos], [cam_colors[c]])

    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        if config['debug_mode']:
            print(glGetError())

        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        move_input(window, delta_time)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)

        square.draw_multiple(depth_program)
        cube.draw_multiple(depth_program)
        for cam in cam_shapes:
            cam.draw_multiple(depth_program)

        hdrbuffer.bind()

        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        glViewport(0, 0, window_width_px, window_height_px)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_objs(square, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)

        update()

        draw_objs(cube, program, perspective, light_pos, texture, normal, specular, depth)
        time.sleep(0.2)

        draw_objs(cube, program, perspective, light_pos, texture, normal, specular, depth)
        for cam in cam_shapes:
            draw_objs(cam, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)

        hdrbuffer.unbind()
        hdrbuffer.finalize()

        bloom.draw_processed_scene()

        glfw.poll_events()
        glfw.swap_buffers(window)

    glfw.terminate()

def update():
    global pressNum, trainedGMMs
    try:
        coord, label = p3.loadCoord(pressNum)


        camera_handles = cv2.VideoCapture("./4persons/video/Take30.59624062.20141124164749.avi")
        fn = 0
        while True:
            ret, image = camera_handles.read()
            if fn == pressNum:
                img = image
                # cv2.imshow('foreground', image)
                # cv2.waitKey(2000)
                break
            fn += 1
        if pressNum == 0:
            cv2.imshow('foreground', image)
            cv2.waitKey(0)
        pathEx = './4persons/extrinsics/Take25.59624062.config.xml'
        fsEx = cv2.FileStorage(pathEx, cv2.FILE_STORAGE_READ)
        mtx = fsEx.getNode('mtx').mat()
        dist = fsEx.getNode('dist').mat()
        rvec = fsEx.getNode('rvec').mat()
        tvec = fsEx.getNode('tvec').mat()

        pts, jac = cv2.projectPoints(np.float32(coord), rvec, tvec, mtx, dist)
        pts = np.int32(pts)
        pixels = []
        for j in range(len(pts)):
            pixels.append(img[pts[j][0][1]][pts[j][0][0]].tolist())
            #cv2.circle(img, tuple([pts[j][0][0], pts[j][0][1]]), 2, img[pts[j][0][1]][pts[j][0][0]].tolist(), -1)

        label = np.squeeze(label)
        pixels = np.array(pixels)
        C0 = pixels[label == 0]
        C1 = pixels[label == 1]
        C2 = pixels[label == 2]
        C3 = pixels[label == 3]
        C2D = [C0, C1, C2, C3]

        coord = np.array(coord)
        Cood0 = coord[label == 0]
        Cood1 = coord[label == 1]
        Cood2 = coord[label == 2]
        Cood3 = coord[label == 3]
        C3D = [Cood0, Cood1, Cood2, Cood3]


        # Euclidean distance of meanValue, no good results
        # GMMs = p3.trainGMM(pressNum)
        # for i in range(len(GMMs)):
        #     distances = [np.linalg.norm(GMMs[i] - mean) for mean in
        #                  [trainedGMMs[0], trainedGMMs[1], trainedGMMs[2], trainedGMMs[3]]]
        #     closest_mean_dis = np.argmin(distances)
        #     print("New data", GMMs[i], "is closest to mean value", closest_mean_dis)

        # predict each pixel, no good results
        # personLabel = []
        # for p in np.array(pixels):
        #     likelihoods = np.zeros(4)
        #     likelihoods[0] = trainedGMMs[0].predict_proba(p.reshape(1, -1))[0, 1]  # likelihood of belonging to person 0
        #     likelihoods[1] = trainedGMMs[1].predict_proba(p.reshape(1, -1))[0, 1]  # likelihood of belonging to person 1
        #     likelihoods[2] = trainedGMMs[2].predict_proba(p.reshape(1, -1))[0, 1]  # likelihood of belonging to person 2
        #     likelihoods[3] = trainedGMMs[3].predict_proba(p.reshape(1, -1))[0, 1]  # likelihood of belonging to person 3
        #     personLabel.append(np.argmax(likelihoods))    # the person with the highest likelihood for this pixel value
        # #print(personLabel)

        predicted_label = []
        for i in range(4):
            likelihoods = [gmm.score(C2D[i]) for gmm in [trainedGMMs[0], trainedGMMs[1], trainedGMMs[2], trainedGMMs[3]]]
            predicted_label.append(likelihoods.index(max(likelihoods)))
            #print(predicted_label)

        color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]   # red 0, green 1, blue 2, yellow 3
        colors = []
        # print(pressNum)
        for j in range(4):
            # print(len(C3D[j]), predicted_label[j], color[predicted_label[j]])
            for C in C3D[j]:
                colors.append(color[predicted_label[j]])
            # print(len(colors))

        position = []
        for C in C3D:
            for item in C:
                position.append(item)

        Rx = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0, -1, 0]])
        positions = [Rx.dot(p) for p in position]
        positions = [np.multiply(DR, 5) for DR in positions]

        NUMBERS = [140,160,170,180,190,
                   200,210,220,260,270,280,290,
                   300,310,320,330,340,350,360,360,380,390,
                   400,410,420,430,440,450,490,
                   500,510,530,530,540,550,560,570,580,590,
                   610,620,630,640,650,660,670,680,690,
                   700,710,720,740,750,780,790,
                   800,810,820,830,840,850,860,870,880,890,
                   900,910,920,930,940,950,960,970,990,
                   1000]
        if pressNum not in NUMBERS:
            cube.set_multiple_positions(positions, colors)
        else:
            pass
        pressNum += 10
    except:
        pass

def resize_callback(window, w, h):
    if h > 0:
        global window_width, window_height, hdrbuffer, blurbuffer
        window_width, window_height = w, h
        glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])
        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        hdrbuffer.delete()
        hdrbuffer.create(window_width_px, window_height_px)
        blurbuffer.delete()
        blurbuffer.create(window_width_px, window_height_px)

def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, glfw.TRUE)
    if key == glfw.KEY_G and action == glfw.PRESS:
        global cube, pressNum

        bg = [f'./4persons/video/Take30.54389819.foreground{pressNum}.jpg',
              f'./4persons/video/Take30.59624062.foreground{pressNum}.jpg',
              f'./4persons/video/Take30.60703227.foreground{pressNum}.jpg',
              f'./4persons/video/Take30.62474905.foreground{pressNum}.jpg']

        #positions, colors = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'], bg, pressNum)


def mouse_move(win, pos_x, pos_y):
    global firstTime, camera, lastPosX, lastPosY
    if firstTime:
        lastPosX = pos_x
        lastPosY = pos_y
        firstTime = False

    camera.rotate(pos_x - lastPosX, lastPosY - pos_y)
    lastPosX = pos_x
    lastPosY = pos_y


def move_input(win, time):
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        camera.move_top(time)
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        camera.move_bottom(time)
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        camera.move_left(time)
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        camera.move_right(time)


if __name__ == '__main__':
    trainedGMMs = p3.trainGMM(pressNum)
    #print(pressNum, trainedGMMs)

    main()
