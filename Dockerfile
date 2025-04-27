FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# install deps + up-to-date CMake (>=3.30)
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget ca-certificates apt-transport-https gnupg build-essential git \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ jammy main" \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir -p build && cd build \
 && cmake .. \
 && make -j$(nproc)

# default: run your convolution binary (override as needed)
CMD ["./build/Problem2/convolution"]
