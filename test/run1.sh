CUDA_VISIBLE_DEVICES=2 nohup python test.py grip apolloscape 0 > log/grip_apolloscape.log &
CUDA_VISIBLE_DEVICES=2 nohup python test.py grip ngsim 0 > log/grip_ngsim.log &
CUDA_VISIBLE_DEVICES=2 nohup python test.py grip nuscenes 0 > log/grip_nuscenes.log &

