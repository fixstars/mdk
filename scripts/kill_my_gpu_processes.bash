#!/bin/bash

# terminate remaining process from outside the main process
# cf. https://github.com/vllm-project/vllm/issues/1908#issuecomment-2114056745

pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

for pid in $pids; do
    user=$(ps -p $pid -o user=)
    if [ "$user" = "$USER" ]; then
        echo "Killing $pid"
        kill -9 $pid
    fi
done
