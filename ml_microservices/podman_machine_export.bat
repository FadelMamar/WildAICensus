
call podman machine stop

call wsl --export podman-machine-default podman.tar

call wsl --unregister podman-machine-default

call wsl --import podman-machine-default D:\PodmanMachine podman.tar

call podman machine start
