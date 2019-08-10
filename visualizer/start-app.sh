#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

SERVER_PATH="~/Desktop/project-NLP-AML/visualizer/server"
FRONTEND_PATH="~/Desktop/project-NLP-AML/visualizer/frontend"

COMPRESSED_MODEL="~/Desktop/drop_bert/model.tar.gz"
DATA_PATH="~/Desktop/project-NLP-AML/data"
PREDICTOR="bert-drop"
CUSTOM_CODE_PATH="~/Desktop/drop-bert/"
CUSTOM_MODULE="drop_bert"

SERVER="python -m server \
--archive-path $COMPRESSED_MODEL \
--data-path $DATA_PATH \
--predictor $PREDICTOR \
--additional-path $CUSTOM_CODE_PATH \
--include-package $CUSTOM_MODULE"

FRONTEND="npm start"

SWITCH_ENV="conda activate allennlp"

if [ -z "$1" ]; then
    tmux new-session -d bash
    tmux split-window -h bash

    tmux send -l -t 0:0.0 "$SWITCH_ENV"
    sleep 2
    tmux send -t 0:0.0 C-m
    tmux send -l -t 0:0.0 "cd $SERVER_PATH"
    tmux send -t 0:0.0 C-m
    sleep 1
    tmux send -l -t 0:0.0 "$SERVER"
    tmux send -t 0:0.0 C-m

    tmux send -l -t 0:0.1 "cd $FRONTEND_PATH"
    tmux send -t 0:0.1 C-m
    sleep 1
    tmux send -l -t 0:0.1 "$FRONTEND"
    tmux send -t 0:0.1 C-m

    tmux -2 attach-session -d
elif [ "$1" = 'server-only' ]; then
    eval "$SWITCH_ENV"
    eval "cd $SERVER_PATH"
    eval "$SERVER"
elif [ "$1" = 'front-only' ]; then
    eval "cd $FRONTEND_PATH"
    eval "$FRONTEND"
fi
