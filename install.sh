# install library dependencies
sudo apt-get install libboost-all-dev libopenblas-dev liblapacke-dev cmake build-essential git
sudo apt-get install python-dev python-pip python-setuptools #python-opencv

# install git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# install the library
pip install --user git+ssh://git@bitbucket.org/macherlabs/facelib.git