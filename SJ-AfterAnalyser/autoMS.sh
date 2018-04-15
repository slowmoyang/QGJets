EVENT_TYPE=$1
CMD="cmsenv && ./merge_and_shuffle ${EVENT_TYPE}"
NAME="MS_${EVENT_TYPE}" # session name
tmux new -s ${NAME} -d
tmux send-keys -t ${NAME} "${CMD}" C-m
