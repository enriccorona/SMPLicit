import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import cv2
import pickle
import torch
import glob
import trimesh
from utils.sdf import create_grid, eval_grid_octree, eval_grid
from utils import projection
from options.image_fitting_options import FitOptions
import kaolin
import time
import tqdm
from utils import image_fitting
import SMPLicit

fitoptions = FitOptions()
_opt = fitoptions.parse()

SMPLicit_Layer = SMPLicit.SMPLicit()
SMPLicit_Layer = SMPLicit_Layer.cuda()

# Initialize SMPL-Related stuff:
SMPL_Layer = SMPLicit_Layer.SMPL_Layer
smpl_faces = torch.LongTensor(SMPL_Layer.faces).cuda()

files = glob.glob(_opt.image_folder + '/*' + _opt.image_extension)
files.sort()

cool_latent_reps = np.load('utils/z_gaussians.npy')

print("PROCESSING:")
print(files)

for _file in files:
    _file = str.split(_file[:-4], '/')[-1]
    path_image = _opt.image_folder + _file + _opt.image_extension
    path_smpl_prediction = _opt.smpl_prediction_folder + _file + '_prediction_result.pkl'
    path_segmentation = _opt.cloth_segmentation_folder + _file + '.png'
    path_instance_segmentation =  _opt.instance_segmentation_folder + _file + '.png'

    input_image = cv2.imread(path_image)
    smpl_prediction = pickle.load(open(path_smpl_prediction, 'rb'))['pred_output_list']

    posed_meshes = []
    posed_normals = []
    colors = []
    all_camscales = []
    all_camtrans = []
    all_toplefts = []
    all_scaleratios = []
    done_people = []
    #for index_fitting in range(1):
    for index_fitting in range(len(smpl_prediction)):
        segmentation = cv2.imread(path_segmentation, 0)
        instance_segmentation = cv2.imread(path_instance_segmentation, 0)
        topleft = smpl_prediction[index_fitting]['bbox_top_left']
        scale_ratio = smpl_prediction[index_fitting]['bbox_scale_ratio']

        offset = 40
        min_y = max(0,int(topleft[1]+offset/scale_ratio))
        max_y = min(input_image.shape[0],int(topleft[1]+(224-offset)/scale_ratio))
        min_x = max(0,int(topleft[0]+offset/scale_ratio))
        max_x = min(input_image.shape[1],int(topleft[0]+(224-offset)/scale_ratio))
        person_instance_segm = instance_segmentation[min_y:max_y,min_x:max_x]
        try:
            index_instancesegm = np.bincount(person_instance_segm.reshape(-1))[1:].argmax() + 1
        except:
            continue

        if index_instancesegm in done_people:
            continue
        done_people.append(index_instancesegm)

        assert index_instancesegm > 0
        segmentation[instance_segmentation!=index_instancesegm] = 0

        # CLEAN NOISE FROM SEGMENTATION:
        if (segmentation == 2).sum() < 100:
            segmentation[segmentation == 2] = 25
        if (segmentation == 5).sum() < 50:
            segmentation[segmentation == 5] = 25
        if (segmentation == 7).sum() < 50:
            segmentation[segmentation == 7] = 25
        if (segmentation == 9).sum() < 50:
            segmentation[segmentation == 9] = 25
        if (segmentation == 12).sum() < 50:
            segmentation[segmentation == 12] = 25
        if (segmentation == 18).sum() < 50:
            segmentation[segmentation == 18] = 25

        # Image crop params:
        camScale = smpl_prediction[index_fitting]['pred_camera'][0] 
        camTrans = smpl_prediction[index_fitting]['pred_camera'][1:]
        topleft = smpl_prediction[index_fitting]['bbox_top_left']
        scale_ratio = smpl_prediction[index_fitting]['bbox_scale_ratio']
        pose = torch.FloatTensor(smpl_prediction[index_fitting]['pred_body_pose'])
        beta = torch.FloatTensor(smpl_prediction[index_fitting]['pred_betas'])

        # Visualization:
        #m = trimesh.Trimesh(smpl_prediction[index_fitting]['pred_vertices_smpl'], smpl_prediction[index_fitting]['faces'])
        #m.show()

        _opt = fitoptions.set_segmentation(segmentation)

        # Get depth image of SMPL, to remove sampling vertices that are further than body: 
        SMPL_Layer = SMPL_Layer.cuda()
        v_posed = SMPL_Layer.forward(beta=beta.cuda(), theta=pose.cuda(), get_skin=True)[0][0].cpu().data.numpy()
        mesh_smpl = trimesh.Trimesh(v_posed, smpl_faces.cpu().data.numpy())
        depth_image_smpl = image_fitting.render_depth_image(mesh_smpl, camScale, camTrans, topleft, scale_ratio, input_image)

        # Prepare final mesh, and we will keep concatenating vertices/faces later on, while updating normals and colors:
        vertices_smpl = SMPL_Layer.forward(beta=beta.cuda(), theta=pose.cuda(), get_skin=True)[0][0].cpu().data.numpy()
        m = trimesh.Trimesh(vertices_smpl, smpl_faces.cpu().data.numpy())
        posed_meshes.append(m)
        posed_normals.append(m.vertex_normals)
        colors.append([220, 220, 220])
        all_camscales.append(camScale)
        all_camtrans.append(camTrans)
        all_toplefts.append(topleft)
        all_scaleratios.append(scale_ratio)

        # Optimizating clothes one at a time
        for cloth_optimization_index in _opt.labels:
            _opt = fitoptions.update_optimized_cloth(cloth_optimization_index)

            # Get SMPL mesh in kaolin:
            #SMPL_Layer = SMPL_Layer.cpu()
            J, v = SMPL_Layer.skeleton(beta.cuda(), require_body=True)
            SMPL_Layer = SMPL_Layer.cuda()
            if _opt.repose:
                v_inference = SMPL_Layer.forward(beta=beta.cuda(), theta=_opt.pose_inference_repose.cuda(), get_skin=True)[0]
            else:
                v_inference = v
            smpl_mesh = kaolin.rep.TriangleMesh.from_tensors(v_inference[0].cuda(), smpl_faces.cuda())
            smpl_mesh_sdf = kaolin.conversions.trianglemesh_to_sdf(smpl_mesh)

            # Sample points uniformly in predefined 3D space of clothing:
            coords, mat = create_grid(_opt.resolution, _opt.resolution, _opt.resolution, np.array(_opt.b_min), np.array(_opt.b_max))
            coords = coords.reshape(3, -1).T
            coords_tensor = torch.FloatTensor(coords)

            # Remove unnecessary points that are too far from body and are never occupied anyway:
            unsigned_distance = torch.abs(smpl_mesh_sdf(coords_tensor.cuda()))
            if cloth_optimization_index == 2:
                valid = unsigned_distance < 0.1
            else:
                #valid = unsigned_distance < 0.001
                valid = unsigned_distance < 0.01
            coords = coords[valid.cpu().data.numpy()]
            coords_tensor = coords_tensor[valid]

            # Re-Pose to SMPL's Image pose:
            if cloth_optimization_index == 9 or cloth_optimization_index == 12:
                unposed_verts = coords_tensor
                model_trimesh = trimesh.Trimesh(unposed_verts, [], process=False)
                model_trimesh = image_fitting.unpose_and_deform_cloth(model_trimesh, _opt.pose_inference_repose.cpu(), pose, beta.cpu(), J, v, SMPL_Layer)
                posed_coords = model_trimesh.vertices
            else:
                # TODO: Move this to image utils script
                SMPL_Layer = SMPL_Layer.cpu()
                posed_coords = np.zeros((len(coords), 3))
                for i in range(len(coords)//_opt.step + 1):
                    unposed_verts = coords_tensor[i*_opt.step:(i+1)*_opt.step]
                    _, batch_unposed_coords = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J.cpu(), v.cpu(), unposed_verts.unsqueeze(0), neighbors=10)
                    posed_coords[i*_opt.step:(i+1)*_opt.step] = batch_unposed_coords[0].cpu().data.numpy()
                SMPL_Layer = SMPL_Layer.cuda()

            # Convert 3D Points to original image space (X,Y are aligned to image)
            coords_2d = image_fitting.project_points(posed_coords, camScale, camTrans, topleft, scale_ratio, input_image)

            # Remove vertices outside of the image
            coords_2d, valid = image_fitting.remove_outside_vertices(coords_2d, input_image)
            coords = coords[valid]
            coords_tensor = coords_tensor[valid]
            posed_coords = posed_coords[valid]

            # Move to image coordinates
            coords_2d = np.int32(coords_2d)[:, :2]

            # Find positive/negative gt:
            dists = np.zeros(len(coords_2d))
            dists[segmentation[coords_2d[:, 1], coords_2d[:, 0]] == 1] = 0

            # TODO: MOVE THIS TO IMAGE UTILS SCRIPT:
            # Find array of indices per pixel:
            array_pixels = []
            array_gt = []
            condys = []
            for y in range(input_image.shape[1]):
                cond2 = coords_2d[:, 0] == y
                condys.append(cond2)
            for x in range(input_image.shape[0]):
                cond1 = coords_2d[:, 1] == x
                # Faster iteration:
                if not cond1.max(): 
                    continue
                for y in range(input_image.shape[1]):
                    cond2 = condys[y]
                    if not cond2.max():
                        continue
                    if instance_segmentation[x, y] == 0 and cloth_optimization_index != 2:
                        continue
                    indices = np.where(np.logical_and(cond1, cond2))[0]
                    if len(indices) == 0:
                        continue

                    depth_smpl = depth_image_smpl[x, y]
                    if depth_smpl == 0:
                        depth_smpl = np.inf

                    indices = indices[coords[indices, 2] < depth_smpl]
                    if len(indices) == 0:
                        continue
                    if segmentation[x, y] in _opt.other_labels:
                        continue

                    array_pixels.append(indices)
                    array_gt.append(_opt.clamp_value - (segmentation[x, y] == cloth_optimization_index)*_opt.clamp_value)

            array_pixels = np.array(array_pixels)
            array_gt = np.array(array_gt)

            if len(array_pixels) < 200:
                continue

            # NOTE: Initialize upper cloth to open jacket's parameters helps convergence when we have detected jacket's segmentation label
            if cloth_optimization_index == 7:
                parameters = image_fitting.OptimizationCloth(_opt.num_params_style, _opt.num_params_shape, cool_latent_reps[3][6:])
            else:
                parameters = image_fitting.OptimizationCloth(_opt.num_params_style, _opt.num_params_shape)
            optimizer = torch.optim.Adam(parameters.parameters(), lr=_opt.lr)

            # Optimization loop:
            # TODO: Add weights to options file:
            weight_occ = 1
            weight_reg = 6
            frames = []
            decay = (_opt.lr - _opt.lr_decayed)/_opt.iterations

            clusters = np.load(SMPLicit_Layer._opt.path_cluster_files + '/' + _opt.clusters, allow_pickle=True)
            for i in tqdm.tqdm(range(_opt.iterations)):
                # Select random number of vertices:
                indices = np.arange(len(array_pixels))
                np.random.shuffle(indices)

                # Get differentiable style and shape vectors:
                style, shape = parameters.forward()

                # Iterate over these samples
                loss = torch.FloatTensor([0]).cuda()
                loss_positives = torch.FloatTensor([0]).cuda()
                loss_negatives = torch.FloatTensor([0]).cuda()
                for index_sample in range(_opt.index_samples):
                    inds_points = array_pixels[indices[index_sample]]
                    gt = array_gt[indices[index_sample]]
                    points = torch.FloatTensor(coords_tensor[inds_points]).cuda()

                    # Positional encoding
                    input_position_points = points[:, None] - v_inference[:, clusters[_opt.num_clusters]].cuda()
                    input_position_points = input_position_points.reshape(1, -1, _opt.num_clusters*3)

                    # Forward pass:
                    if cloth_optimization_index == 18: # Shoe
                        empty_tensor = torch.zeros(1,0).cuda()
                        pred_dists = SMPLicit_Layer.forward(_opt.index_cloth, empty_tensor, style, input_position_points)
                    else:
                        pred_dists = SMPLicit_Layer.forward(_opt.index_cloth, shape, style, input_position_points)

                    # Loss:
                    if gt == _opt.clamp_value:
                        loss = loss + torch.abs(pred_dists - _opt.clamp_value).max()*_opt.weight_negatives
                        loss_negatives = loss_negatives + torch.abs(pred_dists - _opt.clamp_value).max()*_opt.weight_negatives
                    else:
                        lp = torch.min(pred_dists).mean()*_opt.weight_positives

                        loss_positives = loss_positives + lp
                        loss = loss + lp

                # Hyperparameters for fitting T-Shirt and Jacket better. This might require a little bit of tweaking, and challenging images might converge wrongly
                if cloth_optimization_index == 5:
                    reg = torch.abs(style).mean()*10 + torch.abs(shape).mean()
                elif cloth_optimization_index == 7:
                    center_z_style = torch.FloatTensor(cool_latent_reps[3][6:]).cuda()
                    center_z_cut = torch.FloatTensor(cool_latent_reps[3][:6]).cuda()
                    reg = (torch.abs(style - center_z_style).mean() + torch.abs(shape - center_z_cut).mean())*4
                else:
                    reg = torch.abs(style).mean() + torch.abs(shape).mean()

                loss = loss*weight_occ + reg*weight_reg
                # Backward:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for param_group in optimizer.param_groups:
                    param_group['lr'] = _opt.lr - decay*i

            t = time.time()

            # After training, get final prediction:
            style, shape = parameters.forward()

            if cloth_optimization_index == 18: # Shoe
                smpl_mesh, model_trimesh = SMPLicit_Layer.reconstruct([_opt.index_cloth], [style[0]], _opt.pose_inference_repose.cuda(), beta.cuda())
            else:
                smpl_mesh, model_trimesh = SMPLicit_Layer.reconstruct([_opt.index_cloth], [torch.cat((shape, style), 1).cpu().data.numpy()[0]], _opt.pose_inference_repose.cuda(), beta.cuda())

            smooth_normals = model_trimesh.vertex_normals.copy()
            normals = model_trimesh.vertex_normals.copy()

            # Unpose+Pose if it's lower body, and pose directly if it's upper body:
            if cloth_optimization_index == 9 or cloth_optimization_index == 12:
                model_trimesh, normals = image_fitting.unpose_and_deform_cloth_w_normals(model_trimesh, _opt.pose_inference_repose.cpu(), pose, beta.cpu(), J, v, SMPL_Layer, normals)
            else:
                model_trimesh, normals = image_fitting.batch_posing_w_normals(model_trimesh, normals, pose, J, v, SMPL_Layer)

            # Smooth meshes:
            model_trimesh = trimesh.smoothing.filter_laplacian(model_trimesh,lamb=0.5)

            # Save predictions before rendering all of them together
            posed_meshes.append(model_trimesh)
            posed_normals.append(normals)
            colors.append(_opt.color[0][:3])
            all_camscales.append(camScale)
            all_camtrans.append(camTrans)
            all_toplefts.append(topleft)
            all_scaleratios.append(scale_ratio)

    t = time.time()
    original_vertices = []
    for i in range(len(posed_meshes)):
        original_vertices.append(posed_meshes[i].vertices.copy())

    frames = [input_image]*10
    nframes = 100
    angles_y = np.arange(0, 360, 360/nframes)
    diffuminated_image = np.uint8(input_image.copy()*0.4)

    # To avoid person intersections in rendering, just assign person ordering according to person's scale:
    uniqueratios = np.unique(all_scaleratios)
    order = []
    for i in range(len(all_scaleratios)):
        which_smpl = np.where(all_scaleratios[i] == uniqueratios)[0][0]
        order.append(which_smpl/2.)

    # Render frames of persons rotating around Y axis:
    print("Rendering video")
    renderer = image_fitting.meshRenderer()
    for i in tqdm.tqdm(range(len(angles_y))):
        angle_y = angles_y[i]
        c = np.cos(angle_y*np.pi/180)
        s = np.sin(angle_y*np.pi/180)
        rotmat_y = np.array([[c, 0, s], [0, 1, 0], [-1*s, 0, c]])

        for j in range(len(posed_meshes)):
            posed_meshes[j].vertices = np.matmul(rotmat_y, original_vertices[j].copy().T).T
            posed_meshes[j].vertices[:, 2] += order[j]

        renderImg_normals = image_fitting.render_image_projection_multiperson_wrenderer(diffuminated_image, posed_meshes, posed_normals, colors, all_camscales, all_camtrans, all_toplefts, all_scaleratios, mode='normals', renderer=renderer)
        if i == 0:
            for i in range(20):
                frames.append(renderImg_normals)
            diffuminated_image = np.uint8(diffuminated_image*0)
        frames.append(renderImg_normals)

    # Save video:
    print("Saving output video")
    image_fitting.save_video(frames, 'video_fits/%d_result_video'%t, freeze_first=5, freeze_last=5, framerate=24)
