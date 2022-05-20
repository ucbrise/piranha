
nvidia-smi -lms 10 --query-gpu=memory.used --format=csv,noheader

#| awk -F= 'BEGIN { max = -inf } { if ($0 > max) { max = $0; print max } } END { }'

#a=0
#while true; do 
#    b=$(nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')
#    [ $b -gt $a ] && a=$b && echo $a 
#    sleep .1
#done
