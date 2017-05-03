#!/bin/bash

GPU='0'
TAG=''
DIR='./log'
DATA=''
MEM=''
while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -s|--script)
    SCRIPT="$2"
    shift # past argument
    ;;
    -g|--gpu)
    GPU="$2"
    ;;
    -t|--tag)
    TAG=_"$2"
    ;;
    -d|--dir)
    DIR="$2"
    ;;
    -m|--mem)
    MEM="$2"
    ;;
    -mc|--modelclass)
    MODELCLASS="$2"
    ;;
    --data)
    DATA="$2"
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

if [ ${#MEM} -gt 0 ]; then
    MEM="-m $MEM"
fi

DT=`date "+%Y%m%d_%H%M%S"`
SAVDIR=$DIR/${DT}${TAG}
mkdir -p $SAVDIR
cp $SCRIPT $SAVDIR
# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# CPDIR="cp $DIR/data_sep.py $SAVDIR"
# echo $CPDIR
# eval "$CPDIR"

CMD="CUDA_VISIBLE_DEVICES=$GPU nohup python $SCRIPT --train --dt $DT --modelclass $MODELCLASS &> $SAVDIR/log.out &"
echo $CMD
eval "$CMD"
# CUDA_VISIBLE_DEVICES=$GPU nohup python3 $SCRIPT &> $DT &
