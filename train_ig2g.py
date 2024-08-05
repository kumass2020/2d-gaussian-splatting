#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from ig2g.ip2p import InstructPix2Pix
import wandb
from PIL import Image
import numpy as np
from datetime import datetime

# Set up a global variable for date_str to use the same directory in a run
DATE_STR = datetime.now().strftime('%y%H%M')


def save_image_tensor(tensor, iteration, image_name, source_path, base_directory='./output_ig2g'):
    # Extract the scene name from the source path
    scene_name = os.path.basename(source_path.rstrip(os.sep))

    # Ensure the tensor is on the CPU and detached from the computation graph
    tensor = tensor.detach().cpu()

    # Convert tensor to numpy array
    array = tensor.numpy()

    # Convert the array to (H, W, C) format
    array = np.transpose(array, (1, 2, 0))

    # Convert array to uint8 type for image saving
    array = (array * 255).astype(np.uint8)

    # Create an Image object
    image = Image.fromarray(array)

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(base_directory, f'{scene_name}_{DATE_STR}')
    os.makedirs(output_dir, exist_ok=True)

    # Define the image file path
    image_path = os.path.join(output_dir, f'iter{iteration}_{image_name}.png')

    # Save the image
    image.save(image_path)

    print(f"Image saved to {image_path}")


# source_path: dataset.source_path
def render_all_cameras(scene, source_path, gaussians, pipe, background, iteration):
    all_cameras = scene.getTrainCameras()

    print(f"Rendering images for iteration {iteration}...")

    for camera in tqdm(all_cameras, desc="Rendering images from all cameras"):
        render_pkg = render(camera, gaussians, pipe, background)
        image = render_pkg["render"]

        # Use camera.image_name as the image_name
        image_name = camera.image_name

        # Save the rendered image
        save_image_tensor(image, iteration, image_name, source_path)

    print(f"Rendering and saving images for iteration {iteration} complete.")


def clone_edited_images(scene):
    all_cameras = scene.getTrainCameras()

    for camera in tqdm(all_cameras, desc="original image to edited image"):
        camera.edited_image = camera.original_image


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    # Log the ip2p parameters to wandb
    ip2p_params = {
        "ip2p_start_iter": 20000,
        "ip2p_cycle_iter": 2500,
        "guidance_scale": 12.5,
        "image_guidance_scale": 1.5,
        "diffusion_steps": 20,
        "lower_bound": 0.7,
        "upper_bound": 0.98,
        "use_rendered_noise": True,
        # 'None', 'direct', 'normalized', 'tile-normalized',
        # 'direct-encoded', 'normalized-encoded', 'tile-normalized-encoded'
        "noise_type": "direct-encoded",
        "densification_schedule": "normal",
        "ip2p_iter": 3,
    }
    wandb.config.update(ip2p_params)

    ip2p_iteration = 0

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    clone_edited_images(scene)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ip2p = InstructPix2Pix(torch_device, ip2p_use_full_precision=False)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Function to get an identical camera instance from scene.getTrainCameras()
        def get_identical_camera(camera):
            for cam in scene.getTrainCameras():
                if cam == camera:
                    return cam
            return None

        # ip2p
        if (iteration > ip2p_params['ip2p_start_iter']
                and iteration % ip2p_params['ip2p_cycle_iter'] == 1
                and ip2p_iteration < ip2p_params['ip2p_iter']):
            # load base text embedding using classifier free guidance
            text_embedding = ip2p.pipe._encode_prompt(
                "Make it look like it just snowed.",
                device=torch_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=""
            )

            for camera in scene.getTrainCameras():
                render_pkg = render(camera, gaussians, pipe, background)
                image, viewspace_point_tensor, visibility_filter, radii, rendered_noise = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["rend_noise"]
                )

                # Downsample the rendered noise to latent size
                C = image.shape[0]
                H = int(image.shape[1] / 8)
                W = int(image.shape[2] / 8)

                # Process the noise, following noise_type
                if ip2p_params['noise_type'] == "direct":
                    pass
                elif ip2p_params['noise_type'] == "normalized":
                    pass
                elif ip2p_params['noise_type'] == "tile-normalized":
                    pass
                elif ip2p_params['noise_type'] == 'direct-encoded':
                    pass
                elif ip2p_params['noise_type'] == "normalized-encoded":
                    pass
                elif ip2p_params['noise_type'] == "tile-normalized-encoded":
                    pass

                rendered_image = image.unsqueeze(0)
                original_image = camera.original_image.unsqueeze(0)
                rendered_noise = rendered_noise.unsqueeze(0)

                # edit the image using ip2p
                edited_image = ip2p.edit_image(
                    text_embedding.to(torch_device),
                    rendered_image.to(torch_device),
                    original_image.to(torch_device),
                    rendered_noise.to(torch_device),
                    guidance_scale=ip2p_params['guidance_scale'],  # text guidance scale
                    image_guidance_scale=ip2p_params['image_guidance_scale'],
                    diffusion_steps=ip2p_params['diffusion_steps'],
                    lower_bound=ip2p_params['lower_bound'],
                    upper_bound=ip2p_params['upper_bound'],
                    noise_type=ip2p_params['noise_type']
                )

                # resize to original image size (often not necessary)
                if (edited_image.size() != rendered_image.size()):
                    edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:],
                                                                   mode='bilinear')

                # Update edited image
                camera.edited_image = edited_image
                ip2p_iteration += 1

        if iteration in saving_iterations:
            render_all_cameras(scene, dataset.source_path, gaussians, pipe, background, iteration)

        gt_image = viewpoint_cam.edited_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        # Ensure the input tensor is of the same type as the autoencoder's expected input
        gt_image = gt_image.to(image.dtype)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                # Log training metrics to wandb
                wandb.log({
                    "Loss": ema_loss_for_log,
                    "Distortion": ema_dist_for_log,
                    "Normal Loss": ema_normal_for_log,
                    "Iteration": iteration
                })

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                   0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(torch_device), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                                 rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                                 surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                                 rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                                 rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'],
                                                                                         l1_test, psnr_test, ssim_test,
                                                                                         lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                # Log evaluation metrics to wandb
                wandb.log({
                    f"{config['name']} L1 Loss": l1_test,
                    f"{config['name']} PSNR": psnr_test,
                    f"{config['name']} SSIM": ssim_test,
                    f"{config['name']} LPIPS": lpips_test,
                    "Iteration": iteration
                })

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20_000, 27_500, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 27_500, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize wandb and log all arguments
    wandb.init(project="2DGS-InstructGaussians2Gaussians")
    wandb.config.update(args)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
