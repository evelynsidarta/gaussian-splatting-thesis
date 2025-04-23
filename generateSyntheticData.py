import bpy
import math
import string
from math import radians
# from mathutils import Matrix
from mathutils import Vector
import json
import random
import os

# input variables, change depending on needs
numHeight = 6                         # determines how many different height levels to capture
numCapturesPerHeight = 20               # determines how many different captures per height level
BBoxObj = {
    'Min' : [-2.0, -1.0, 0.0],          # rough bottom left point of the object
    'Max' : [3.0, 1.0, 6.0]             # end point of the box (opposite end of startPoint
}

# multiplier for the maximum distance
Multiplier = 2.5
IgnoreDegrees = 60.0

# Blender camera's field of view:
#   if image is square, hFoV = vFoV
#   if image is rectangular and height of image > width, then fov of camera is the vFov (vertical)
#   if image is rectangular and width of image > height, then fov of camera is the hFoV (horizontal)
#   default camera fov in blender: 50.0
#   fov converter online: https://phrogz.net/tmp/fov-converter.html#x:1920_y:1080_h:50.0_v:29.39_key:h
FoV = 50.0

# other inputs to adjust based on needs
NumTrn = numHeight * numCapturesPerHeight
# center of the object = middle point between min and max points of the object bounding box
ObjCenter = [
    ((BBoxObj['Min'][0] + BBoxObj['Max'][0]) / 2.0),
    ((BBoxObj['Min'][1] + BBoxObj['Max'][1]) / 2.0),
    ((BBoxObj['Min'][2] + BBoxObj['Max'][2]) / 2.0)
]
# Radius of 'circles'
xRadius = BBoxObj['Max'][0] - ObjCenter[0]
yRadius = BBoxObj['Max'][1] - ObjCenter[1]
zRadius = BBoxObj['Max'][2] - ObjCenter[2]

# calculate horizontal and vertical step
hStep = 0.0
vStep = 0.0
if (numCapturesPerHeight > 0):
    hStep = 360.0 / numCapturesPerHeight
if (numHeight > 0):
    # to avoid taking pictures from direct z-axis
    vStep = (180.0 - IgnoreDegrees) / (numHeight + 1)

# set paths
TrnPath = 'C:/Uni/Thesis/blender/output/train/'
ValPath = 'C:/Uni/Thesis/blender/output/val/'
TstPath = 'C:/Uni/Thesis/blender/output/test/'
OutPath = 'C:/Uni/Thesis/blender/output/'

# setup
def setupRenderSettings(vOutputPath):
    # assign current scene in Blender to sc
    sc = bpy.context.scene
    # use_nodes is a boolean that connects inputs with outputs, used to represent processes in Blender
    sc.use_nodes = True
    tr = sc.node_tree
    # clear tree nodes
    tr.nodes.clear()
    
    # add new compositor node to the tree
    # CompositorNodeRLayers represents output of a render layer, provides access to rendered image, alpha channel, depth, etc.
    RenderLayers = tr.nodes.new(type = "CompositorNodeRLayers")
    
    # CompositorNodeOutputFile allows to save data from compositor to external files
    # used to save individual render passes, customize outputs
    OutputImage = tr.nodes.new(type = "CompositorNodeOutputFile")
    OutputImage.format.file_format = 'PNG'
    # base_path specifies the directory where files will be saved
    OutputImage.base_path = vOutputPath
    
    # create another node for the depth map
    OutputDepth = tr.nodes.new(type = "CompositorNodeOutputFile")
    OutputDepth.format.file_format = 'OPEN_EXR'
    OutputDepth.base_path = vOutputPath
    
    # send image information from RenderLayers to OutputImage's first input socket
    tr.links.new(RenderLayers.outputs["Image"], OutputImage.inputs[0])
    # send depth information from RenderLayers to OutputDepth's first input socket
    tr.links.new(RenderLayers.outputs["Depth"], OutputDepth.inputs[0])
    
    return OutputImage, OutputDepth

def generate_random_string(length):
    pool = string.ascii_letters + string.digits
    rd = ''
    for i in range(length):
      rd = rd + random.choice(pool)
    print(rd + "with length " + str(length))
    return rd

def generate_camera_position(height, captureNo, maxDistance):
    
    # 1. translate everything to origin (0, 0, 0)
    # 2. rotate around the y-axis according to vStep
    # 3. rotate around the z-axis according to hStep
    
    # calculate rotation values around y- and z-axes
    vRot = (vStep * (height + 1)) + IgnoreDegrees   # since height value starts at 0
    hRot = hStep * (captureNo)
    
    # initial: find spot at the z-axis
    xTemp = 0.0
    yTemp = 0.0
    zTemp = maxDistance
    xPos = 0.0
    yPos = 0.0
    zPos = 0.0
    
    # rotate vStep degrees around the y-axis
    # x' = z * sin(vStep) + x * cos(vStep)
    # z' = z * cos(vStep) - x * sin(vStep)
    xTemp = zTemp * math.sin(math.radians(vRot)) # second half can be omitted since initial x value is always 0
    zTemp = zTemp * math.cos(math.radians(vRot)) # second half can be omitted since initial x value is always 0
    
    # rotate hStep degrees around the z-axis
    # x' = x * cos(hStep) - y * sin(hStep)
    # y' = x * sin(hStep) + y * cos(hStep)
    xPos = xTemp * math.cos(math.radians(hRot)) # second half can be skipped because initial y value is always 0
    yPos = xTemp * math.sin(math.radians(hRot)) # second half can be skipped because initial y value is always 0
    zPos = zTemp
    
    # translate back according to the object center
    xPos = xPos + ObjCenter[0]
    yPos = yPos + ObjCenter[1]
    zPos = zPos + ObjCenter[2]
    return (xPos, yPos, zPos)

def lookAt(Camera, LookAtPosition):
    Direction = LookAtPosition - Camera.location
    Rotation_quat = Direction.to_track_quat('-Z', 'Y')
    Camera.rotation_euler = Rotation_quat.to_euler()

def update_camera_params(CameraPosition, LookAtPosition):
    Camera = bpy.context.scene.camera
    Camera.location = CameraPosition
    # tell camera to look at LookAtPosition
    lookAt(Camera, LookAtPosition)
    bpy.context.view_layer.update()
    FoVx = bpy.context.scene.camera.data.angle_x
    transform_matrix = bpy.context.scene.camera.matrix_world
    transform_matrix_list = [list(row) for row in transform_matrix]
    CameraParams = {
        "camera_angle_x" : FoVx,
        "transform_matrix" : transform_matrix_list
    }
    return CameraParams

# TODO
# params list:
#               1. CameraCenter - location of camera
#               2. CameraPosition
def renderSurround(CameraCenter, CameraLookAt, CameraParams, OutputImage, OutputDepth):
    # if there is nothing filled as camera center
    # i - counter for image
    i = 0
    
    if (len(CameraCenter) == 0):
        # loop through all intended captures:
        # vertical change   : numHeight, 
        # horizontal change : numCapturesPerHeight
        maxRad = max(xRadius, yRadius, zRadius)
        alpha = math.sin(FoV)
        maxDist = Multiplier * (maxRad / alpha) * math.sqrt(1 - (alpha * alpha))
        
        for h in range(numHeight):
            for n in range(numCapturesPerHeight):
                # TODO:
                # 1. customize camPosTemp and LookAtPosition so object is always in frame
                #
                CamPosTemp = generate_camera_position(h, n, maxDist)     # already including offset
                CameraCenter.append(CamPosTemp)
                # camera is always looking at the center of the object
                LookAtPosition = (ObjCenter[0], ObjCenter[1], ObjCenter[2])
                CameraLookAt.append(LookAtPosition)
                # TODO: update camera params, customize the parameters of the function
                CameraParamsTemp = update_camera_params(Vector(CamPosTemp), Vector(LookAtPosition))
                CameraParams.append(CameraParamsTemp)

                CurRenderFrame = str(i)
                OutputImage.file_slots[0].path = f"image{CurRenderFrame}"
                OutputDepth.file_slots[0].path = f"depth{CurRenderFrame}"
                bpy.ops.render.render(write_still = True)
            
                # update counter
                i = i + 1
                
    elif (len(CameraParams) == 0):
        for h in range(numHeight):
            for n in range(numCapturesPerHeight):
                # camera center file is not empty, we retrieve data from this file
                CamPosTemp = Vector(CameraCenter[h * n])
                LookAtPosition = Vector(CameraLookAt[h * n])
                CameraParamsTemp = update_camera_params(CamPosTemp, LookAtPosition)
                CameraParams.append(CameraParamsTemp)
            
                CurRenderFrame = str(i)
                OutputImage.file_slots[0].path = f"image{CurRenderFrame}"
                OutputDepth.file_slots[0].path = f"depth{CurRenderFrame}"
                bpy.ops.render.render(write_still = True)
            
                # update counter
                i = i + 1
                
    else:
        # if all data provided, simply retrieve the data from the files
        for h in range(numHeight):
            for n in range(numCapturesPerHeight):
                # TODO: adjust code
                CamPosTemp = Vector(CameraCenter[h * n])
                LookAtPosition = Vector(CameraLookAt[h * n])
                CameraParamsTemp = update_camera_params(CamPosTemp, LookAtPosition)
    
                CurRenderFrame = str(i)
                OutputImage.file_slots[0].path = f"image{CurRenderFrame}"
                OutputDepth.file_slots[0].path = f"depth{CurRenderFrame}"
                bpy.ops.render.render(write_still = True)
            
                # update counter
                i = i + 1
                
# run
# if nothing to train, do not start
if (NumTrn > 0):
    # setup output files for images and depth map
    OutputImage, OutputDepth = setupRenderSettings(TrnPath)
    
    CameraParams = []
    CameraCenter = []
    CameraLookAt = []
    
    # generate random unique string as identifier
    rd = generate_random_string(6)
    
    # load camera_center and camera_lookat files if it exists
    if (os.path.isfile(f'{OutPath}/camera_center.json')):
        with open(f'{OutPath}/camera_center.json', 'r') as CameraCenterFile:
            CameraCenter = json.load(CameraCenterFile)
        with open(f'{OutputPath}/camera_lookat.json', 'r') as CameraLookAtFile:
            CameraLookAt = json.load(CameraLookAtFile)
    
    OutputImage.base_path = TrnPath + rd + '/train/'
    OutputDepth.base_path = TrnPath + rd + '/depth/'
    TempPath = TrnPath + rd + '/'
        
    # methodically take surround viewpoints to render the object
    ############# TODO: change params
    # params:
    #           CameraDatasTrain - initialized empty
    #           CameraCenter     - initialized empty
    #           CameraDatasTrain - initialized empty
    renderSurround(CameraCenter, CameraLookAt, CameraParams, OutputImage, OutputDepth) # TODO!)
        
    # generate cameraLookAt, cameraCenter, and CameraDatasTrain file
    with open(f'{TempPath}transforms_train.json', 'w') as f:
        json.dump(CameraParams, f, indent = 4)
    with open(f'{TempPath}camera_center.json', 'w') as f:
        json.dump(CameraCenter, f, indent = 4)
    with open(f'{TempPath}camera_lookat.json', 'w') as f:
        json.dump(CameraLookAt, f, indent = 4)
