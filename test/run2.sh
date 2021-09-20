CUDA_VISIBLE_DEVICES=1 nohup python test.py fqa apolloscape 0 > log/fqa_apolloscape.log &
CUDA_VISIBLE_DEVICES=1 nohup python test.py fqa ngsim 0 > log/fqa_ngsim.log &
CUDA_VISIBLE_DEVICES=1 nohup python test.py fqa nuscenes 0 > log/fqa_nuscenes.log &
CUDA_VISIBLE_DEVICES=2 nohup python test.py trajectron apolloscape 0 > log/trajectron_apolloscape.log &
CUDA_VISIBLE_DEVICES=3 nohup python test.py trajectron ngsim 0 > log/trajectron_ngsim.log &
CUDA_VISIBLE_DEVICES=3 nohup python test.py trajectron nuscenes 0 > log/trajectron_nuscenes.log &
CUDA_VISIBLE_DEVICES=3 nohup python test.py trajectron_map nuscenes 0 > log/trajectron_map_nuscenes.log &
