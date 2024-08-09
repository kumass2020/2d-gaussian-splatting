CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_2.sh > ./log/240809_1000_2.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_3.sh > ./log/240809_1000_3.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=4 nohup bash scripts/run_4.sh > ./log/240809_1000_4.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=5 nohup bash scripts/run_5.sh > ./log/240809_1000_5.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=6 nohup bash scripts/run_6.sh > ./log/240809_1000_6.log 2>&1 &
sleep 2
CUDA_VISIBLE_DEVICES=7 nohup bash scripts/run_7.sh > ./log/240809_1000_7.log 2>&1 &
