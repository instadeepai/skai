# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to generate docker build instructions."""

GPU_ACCELERATORS = ['P100', 'V100', 'P4', 'T4', 'A100']
TPU_ACCELERATORS = ['TPU_V2', 'TPU_V3']

CPU_BASE_IMAGE = 'tensorflow/tensorflow:2.13.0'
GPU_BASE_IMAGE = 'nvcr.io/nvidia/tensorflow:23.08-tf2-py3'
TPU_BASE_IMAGE = 'ubuntu:20.04'


def tpuvm_docker_instructions() -> list[str]:
  """Returns a list of docker instructions necessary to use TF 2.9.1 on TPUs."""
  tpu_shared_object_url = 'https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.3.0/libtpu.so'
  tf_wheel_url = 'https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.9.1/tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl'
  return [
      f'RUN wget {tpu_shared_object_url} -O /lib/libtpu.so',
      'RUN chmod 700 /lib/libtpu.so',
      f'RUN wget {tf_wheel_url}',
      'RUN pip3 install tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl',
      'RUN rm tensorflow-2.9.1-cp38-cp38-linux_x86_64.whl',
  ]


def get_docker_instructions(accelerator: str) -> tuple[str, list[str]]:
  """Returns the docker instructions and base image for `accelerator`."""
  if accelerator in TPU_ACCELERATORS:
    # Required by TPU vm.
    base_image = TPU_BASE_IMAGE
    # Build can get stuck for a while without this line.
    docker_instructions = [
        'ENV DEBIAN_FRONTEND=noninteractive',
    ]
    # Make sure python executable is python3.
    docker_instructions += [
        'RUN apt-get update && apt-get install -y python3-pip wget'
    ]
    docker_instructions += tpuvm_docker_instructions()

  elif accelerator in GPU_ACCELERATORS:
    # Select a base GPU image. Other options can be found in
    # https://cloud.google.com/deep-learning-containers/docs/choosing-container
    base_image = GPU_BASE_IMAGE
    docker_instructions = [
        # Add deadsnakes repo
        'RUN apt update',
        'RUN apt-get install software-properties-common -y',
        'RUN add-apt-repository ppa:deadsnakes/ppa -y',

        # Install Python 3.10',
        'RUN apt update && apt install -y python3.10 python3.10-distutils',
        'RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10',

        # Replace python shell with python3.10',
        'RUN unlink /usr/bin/python',
        'RUN ln -s /usr/bin/python3.10 /usr/bin/python',

        'RUN python -m pip install --pre --extra-index-url ' +
        'https://developer.download.nvidia.com/compute/redist/jp/v50 ' +
        'tensorflow==2.13'
    ]

  else:
    # Select a base CPU image. Other options can be found in
    # https://cloud.google.com/deep-learning-containers/docs/choosing-container
    base_image = CPU_BASE_IMAGE
    docker_instructions = [
        'RUN apt-get update && apt-get install -y python3-pip wget',
    ]
  docker_instructions += [
      'RUN apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev '
      'libglib2.0-0 python-is-python3'
  ]
  docker_instructions += [
      'WORKDIR /skai',
      'COPY skai/requirements.txt /skai/requirements.txt',
      'RUN pip install --upgrade pip',
      'RUN pip install --timeout 1000 -r requirements.txt',
      'COPY skai/ /skai',
  ]

  return base_image, docker_instructions
