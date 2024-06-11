#!/bin/bash

docker run --rm --gpus '"device=0"' -it --shm-size=100g --ulimit memlock=-1 --ulimit stack=67108864 -v /data:/data -v /home/$USER:/home/$USER --entrypoint /bin/bash $1

# use alignment-handbook-mihir