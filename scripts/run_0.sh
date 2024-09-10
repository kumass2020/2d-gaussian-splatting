python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/bicycle --eval --port 8820 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-003527/chkpnt20000.pth --noise_type "encoded-normalized" --noise_reg "outlier" --original_caption "A photo of bicycle" --text_prompt "Make it look like it just snowed." --modified_caption "A photo of bicycle, which looks like it just snowed." --is_freeu 1 --freeu_mode "intermediate" --freeu_s1 0.9 --freeu_s2 0.2 --freeu_b1 1.2 --freeu_b2 1.4

python train_ig2g.py -s ../orig/gaussian-splatting/download/mip-nerf-360/360_v2/bonsai --eval --port 8820 --depth_ratio 0 --iteration 40_000 --position_lr_max_steps 40_000 --guidance_scale 12.5 --image_guidance_scale 1.5 --resolution 8 --start_checkpoint output/240809-005034/chkpnt20000.pth --noise_type "encoded-normalized" --noise_reg "outlier" --original_caption "A photo of bonsai" --text_prompt "Make it look like it just snowed." --modified_caption "A photo of bonsai, which looks like it just snowed." --is_freeu 1 --freeu_mode "intermediate" --freeu_s1 0.9 --freeu_s2 0.2 --freeu_b1 1.2 --freeu_b2 1.4