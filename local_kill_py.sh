ps -U carmonda | grep 'python' | awk '{print $1}' | xargs -i ps -f -p {}
ps -U carmonda | grep 'python' | awk '{print $1}' | xargs -t kill -9
