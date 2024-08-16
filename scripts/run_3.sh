python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/garden --eval --port 8823 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-004707/chkpnt20000.pth --noise_type "encoded-normalized" --noise_reg "scaling" --original_caption "A photo of garden" --text_prompt "Make it look like it just snowed." --modified_caption "A photo of garden, which looks like it just snowed."

python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/kitchen --eval --port 8823 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-003543/chkpnt20000.pth --noise_type "encoded-normalized" --noise_reg "scaling" --original_caption "A photo of kitchen" --text_prompt "Turn the lego into white." --modified_caption "A photo of kitchen, which includes white lego."