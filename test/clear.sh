pids=$(ps -aux | grep qzzhang | grep "python\ test\.py" | awk '{print $2}')
kill $pids
