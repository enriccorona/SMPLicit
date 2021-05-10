import numpy as np
import torch
import pyrender

def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    data3D = data3D.copy()
    resnet_input_size_half = 224 *0.5
    if bAppTransFirst:      # Hand model
        data3D[:,0:2] += trans
        data3D *= scale   # apply scaling
    else:
        data3D *= scale # apply scaling
        data3D[:,0:2] += trans

    data3D = data3D*resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    #data3D*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    # data3D[:,:2]*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    return data3D

def convert_smpl_to_bbox_tensor(data3D, scale, trans, bAppTransFirst=False):
    data3D = data3D.clone()
    resnet_input_size_half = 224 *0.5
    if bAppTransFirst:      # Hand model
        data3D[:,0:2] = data3D[:, 0:2] + torch.FloatTensor(trans).cuda()
        data3D *= scale   # apply scaling
    else:
        data3D *= scale # apply scaling
        data3D[:,0:2] = data3D[:, 0:2] + torch.FloatTensor(trans).cuda()

    data3D = data3D*resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    #data3D*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    # data3D[:,:2]*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    return data3D

def convert_smpl_to_bbox_tensor2(data3D, scale, trans, bAppTransFirst=False):
    data3D = data3D.clone()
    resnet_input_size_half = 224 *0.5
    if bAppTransFirst:      # Hand model
        data3D[:,0:2] = data3D[:, 0:2] + trans
        data3D *= scale   # apply scaling
    else:
        data3D *= scale # apply scaling
        data3D[:,0:2] = data3D[:, 0:2] + trans

    data3D = data3D*resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    #data3D*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    # data3D[:,:2]*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    return data3D

def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    data3D = data3D.copy()
    resnet_input_size_half = 224 *0.5
    imgSize = np.array([imgSizeW,imgSizeH])

    data3D /= boxScale_o2n

    data3D[:,:2] += (np.array(bboxTopLeft) + resnet_input_size_half/boxScale_o2n)

    return data3D

def convert_bbox_to_oriIm_tensor(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    try:
        data3D = data3D.copy()
    except:
        data3D = data3D.clone()

    resnet_input_size_half = 224 *0.5
    imgSize = np.array([imgSizeW,imgSizeH])

    data3D /= boxScale_o2n

    add = torch.FloatTensor(np.array(bboxTopLeft) + resnet_input_size_half/boxScale_o2n).cuda()
    data3D[:,:2] = data3D[:, :2] + add
    #data3D[:,:2] += (np.array(bboxTopLeft) + resnet_input_size_half/boxScale_o2n)
    return data3D

def convert_bbox_to_oriIm_tensor2(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    try:
        data3D = data3D.copy()
    except:
        data3D = data3D.clone()

    resnet_input_size_half = 224 *0.5
    imgSize = np.array([imgSizeW,imgSizeH])

    data3D /= boxScale_o2n

    add = torch.FloatTensor(np.array(bboxTopLeft) + resnet_input_size_half/boxScale_o2n).cuda()
    data3D[:,:2] = data3D[:, :2] + add
    #data3D[:,:2] += (np.array(bboxTopLeft) + resnet_input_size_half/boxScale_o2n)
    return data3D

def render_mesh(model):
    dist=2; angle=20; height=-0.2
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=(0, 0, 0))
    scene.add(pyrender.Mesh.from_trimesh(model, smooth=False))
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
    scene.add(light, pose=np.eye(4))
    c = np.cos(angle*np.pi/180)
    s = np.sin(angle*np.pi/180)
    camera_pose = np.array([[c, 0, s, dist*s],[0, 1, 0, height],[-1*s, 0, c, dist*c],[0, 0, 0, 1]])
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, znear=0.5, zfar=5)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(512, 512)
    color, _ = renderer.render(scene)

    return color[:, :, ::-1]

