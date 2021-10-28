mode=$1

nohup python test.py fqa apolloscape 0 ${mode} > log/fqa_apolloscape_${mode}.log &
# nohup python test.py fqa ngsim 0 > log/fqa_ngsim.log &
# nohup python test.py fqa nuscenes 0 > log/fqa_nuscenes.log &
nohup python test.py trajectron apolloscape 0 ${mode} > log/trajectron_apolloscape_${mode}.log &
# nohup python test.py trajectron ngsim 0 > log/trajectron_ngsim.log &
# nohup python test.py trajectron nuscenes 0 > log/trajectron_nuscenes.log &
# nohup python test.py trajectron_map nuscenes 0 > log/trajectron_map_nuscenes.log &
