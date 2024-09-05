CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_1.sh > ./log/240904_1600_1.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_2.sh > ./log/240816_1600_2.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_3.sh > ./log/240816_1600_3.log 2>&1 &