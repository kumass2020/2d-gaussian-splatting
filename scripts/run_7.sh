python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/treehill --eval --port 8827 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-004707/chkpnt20000.pth --noise_type "direct-encoded-normalized" --original_caption "A photo of treehill" --text_prompt "Make it look like it just snowed." --modified_caption "A photo of treehill, which looks like it just snowed."