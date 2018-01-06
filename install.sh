#!/bin/bash

# install library dependencies
sudo apt-get install libboost-all-dev libopenblas-dev liblapacke-dev cmake build-essential git wget
sudo apt-get install python-dev python-pip python-setuptools #python-opencv

# install git lfs
if [ $(uname -m) == "x86_64" ]; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
elif [ $(uname -m) == "armv7l" ]; then
    wget https://dl.google.com/go/go1.9.2.linux-armv6l.tar.gz
    tar -xzf go1.9.2.linux-armv6l.tar.gz
    export GOROOT=$HOME/go
    export PATH=$GOROOT/bin:$PATH
    export GOPATH=$HOME
    export PATH=$GOPATH/bin:$PATH
    go get github.com/github/git-lfs
fi
git lfs install

# install the library
pip install --user git+ssh://git@bitbucket.org/macherlabs/facelib.git