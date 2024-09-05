python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/room --eval --port 8823 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-004320/chkpnt20000.pth --noise_type "None" --noise_reg "None" --original_caption "A photo of room" --text_prompt "Turn the floor into white." --modified_caption "A photo of room, which has white floor."

python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/stump --eval --port 8823 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-003548/chkpnt20000.pth --noise_type "None" --noise_reg "None" --original_caption "A photo of stump" --text_prompt "Make it look like it just snowed." --modified_caption "A photo of stump, which looks like it just snowed."

python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/treehill --eval --port 8823 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-004750/chkpnt20000.pth --noise_type "None" --noise_reg "None" --original_caption "A photo of treehil" --text_prompt "Make it look like it just snowed." --modified_caption "A photo of treehill, which looks like it just snowed."