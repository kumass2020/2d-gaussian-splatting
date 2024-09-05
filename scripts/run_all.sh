CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_0.sh > ./log/240905_1300_0.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_3.sh > ./log/240905_1300_3.log 2>&1 &