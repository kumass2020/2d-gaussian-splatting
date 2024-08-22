import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d
import datetime

import os
import sys
from argparse import ArgumentParser, Namespace


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    combined_args_list = []

    if hasattr(args_cmdline, 'model_dirs'):
        # Process each directory specified in model_dirs
        for model_dir in args_cmdline.model_dirs:
            cfgfile_string = "Namespace()"
            cfgfilepath = os.path.join(model_dir, "cfg_args")
            print(f"Looking for config file in {cfgfilepath}")

            try:
                with open(cfgfilepath) as cfg_file:
                    print(f"Config file found: {cfgfilepath}")
                    cfgfile_string = cfg_file.read()
            except FileNotFoundError:
                print(f"Config file not found at {cfgfilepath}")

            # Parse the cfg_args file as a Namespace
            args_cfgfile = eval(cfgfile_string)

            # Merge command-line arguments with config file arguments
            merged_dict = vars(args_cfgfile).copy()
            for k, v in vars(args_cmdline).items():
                if v is not None:
                    merged_dict[k] = v

            # Create a Namespace for the merged arguments and add to the list
            combined_args = Namespace(**merged_dict)
            combined_args_list.append(combined_args)

    else:
        # Fallback to original single model_path handling if model_dirs is not set
        cfgfile_string = "Namespace()"
        try:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            print(f"Looking for config file in {cfgfilepath}")
            with open(cfgfilepath) as cfg_file:
                print(f"Config file found: {cfgfilepath}")
                cfgfile_string = cfg_file.read()
        except (TypeError, FileNotFoundError):
            print("Config file not found at", cfgfilepath)
            pass

        args_cfgfile = eval(cfgfile_string)

        merged_dict = vars(args_cfgfile).copy()
        for k, v in vars(args_cmdline).items():
            if v is not None:
                merged_dict[k] = v

        combined_args = Namespace(**merged_dict)
        combined_args_list.append(combined_args)

    return combined_args_list


def get_timestamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M")

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("-md", "--model_dirs", nargs='+', help="List of model directories")

    # Get combined arguments for all directories
    combined_args_list = get_combined_args(parser)

    timestamp = get_timestamp()

    for args in combined_args_list:
        model_dir = args.model_path

        # Extract the last part of the source path
        if hasattr(args, 'source_path'):
            source_name = os.path.basename(args.source_path.rstrip('/'))
        else:
            source_name = os.path.basename(model_dir.rstrip('/'))

        output_dir = os.path.join('output_render/'+timestamp, source_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Rendering {args.model_path} into {output_dir}")

        dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_dir = os.path.join(output_dir, 'train', f"ours_{scene.loaded_iter}")
        test_dir = os.path.join(output_dir, 'test', f"ours_{scene.loaded_iter}")
        traj_dir = os.path.join(output_dir, 'traj', f"ours_{scene.loaded_iter}")
        gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

        if not args.skip_train:
            print("Exporting training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir)

        if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
            print("Exporting rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)

        if args.render_path:
            print("Rendering videos ...")
            os.makedirs(traj_dir, exist_ok=True)
            n_frames = 240
            cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
            gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.export_image(traj_dir)
            create_videos(base_dir=traj_dir,
                          input_dir=traj_dir,
                          out_name='render_traj',
                          num_frames=n_frames)

        if not args.skip_mesh:
            print("Exporting mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.gaussians.active_sh_degree = 0
            gaussExtractor.reconstruction(scene.getTrainCameras())

            if args.unbounded:
                name = 'fuse_unbounded.ply'
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

            mesh_path = os.path.join(train_dir, name)
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Mesh saved at {mesh_path}")

            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            mesh_post_path = os.path.join(train_dir, name.replace('.ply', '_post.ply'))
            o3d.io.write_triangle_mesh(mesh_post_path, mesh_post)
            print(f"Mesh post-processed saved at {mesh_post_path}")

