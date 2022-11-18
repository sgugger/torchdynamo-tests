# Installation guide on a new instance

Jump to the last section if you alrady have CUDA installed.

## Install drivers:

```bash
sudo apt install ubuntu-drivers-common
```

Run

```bash
ubuntu-drivers devices
```

Output

```
== /sys/devices/pci0000:00/0000:00:04.0==
modalias : pci:v000010DEd000020B0sv000010DEsd0000134Fbc03sc02i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-470-server - distro non-free
driver   : nvidia-driver-515-open - distro non-free recommended
driver   : nvidia-driver-515 - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-510 - distro non-free
driver   : nvidia-driver-510-server - distro non-free
driver   : nvidia-driver-515-server - distro non-free
driver   : nvidia-driver-470 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin 
```

Pick number recommended

```
sudo apt install nvidia-headless-515-server nvidia-utils-515-server
```

Reboot
```bash
sudo reboot
```

## Install CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

Add public key/repo
```bash
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

Install cuda toolkit. 
If you want to use conda setup, please refer the corresponding section and skip below steps.

```bash
sudo apt update
sudo apt install cuda-toolkit-11-7
```

(you can type-hint after cuda-toolkit- to find all available versions.)

Download [CUDNN](https://developer.nvidia.com/cudnn) and scp it to the instance.

Extract

```bash
tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Add to .bashrc

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/:$PATH"
```

then

```bash
source ~/.bashrc
```

Check everything is alright

```bash
nvidia-smi
```

## Python

```bash
sudo apt-get install pip
sudo apt install python3.8-venv
python3 -m venv dynamo
```

Add to .bashrc
```bash
source dynamo/bin/activate
```

then

```bash
source ~/.bashrc
```

Install nightlies with dynamo

```bash
pip install numpy
pip install --pre torch[dynamo] --extra-index-url https://download.pytorch.org/whl/nightly/cu117/
```

# Conda Installation instructions

1. Install miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. Create conda python env and then activate it
```bash
conda create --name dynamo python
conda activate dynamo
```

3. Install cudatoolkit-11.7. Please refer [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit)
for more information
```bash
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```

4. Verify torch-dynamo with below command assuming you are in the top folder
```
python tools/verify_dynamo.py
```