from typing import List, Tuple

GPU_ACCELERATORS = ["P100", "V100", "P4", "T4", "A100"]
TPU_ACCELERATORS = ["TPU_V2", "TPU_V3"]

CPU_BASE_IMAGE = "tensorflow/tensorflow:2.13.0"
GPU_BASE_IMAGE = "tensorflow/tensorflow:2.13.0-gpu"
TPU_BASE_IMAGE = "ubuntu:22.04"


def tpuvm_docker_instructions() -> List[str]:
    """Returns a list of docker instructions necessary to use TF 2.12.0 on TPUs."""
    return [
       'FROM python:3.8',
       'RUN pip3 install '
       'https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.12.0/tensorflow-2.12.0-cp38-cp38-linux_x86_64.whl',
       'RUN curl -L https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.6.0/libtpu.so -o /lib/libtpu.so'
    ]

def get_docker_instructions(accelerator: str) -> Tuple[str, List[str]]:
    """Returns the required docker instructions and base image for `accelerator`."""
    if accelerator in TPU_ACCELERATORS:
        # Required by TPU vm.
        base_image = TPU_BASE_IMAGE
        # Build can get stuck for a while without this line.
        docker_instructions = [
            "ENV DEBIAN_FRONTEND=noninteractive",
        ]
        # Make sure python executable is python3.
        docker_instructions += [
            "RUN apt-get update && apt-get install -y python3-pip wget"
        ]
        docker_instructions += tpuvm_docker_instructions()

    elif accelerator in GPU_ACCELERATORS:
        # Select a base GPU image. Other options can be found in
        # https://cloud.google.com/deep-learning-containers/docs/choosing-container
        base_image = GPU_BASE_IMAGE
        # Make sure python executable is python3.
        docker_instructions = [
            "RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/"
            "compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub",
            "RUN apt-get update && apt-get install -y python3-pip wget",
        ]

    else:
        # Select a base CPU image. Other options can be found in
        # https://cloud.google.com/deep-learning-containers/docs/choosing-container
        base_image = CPU_BASE_IMAGE
        docker_instructions = [
            "RUN apt-get update && apt-get install -y python3-pip wget",
        ]
        docker_instructions += [
            "RUN apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev "
            "libglib2.0-0 python-is-python3"
        ]
    docker_instructions += [
        "WORKDIR /skai",
        "COPY skai/requirements.txt /skai/requirements.txt",
        "RUN pip3 install --upgrade pip",
        "RUN pip3 install --timeout 1000 -r requirements.txt",
        "COPY skai/ /skai",
    ]

    return base_image, docker_instructions