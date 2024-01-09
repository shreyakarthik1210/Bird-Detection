Bird Identification using tflite, raspberry pi, and the pi camera module. 

1. Set up raspberry pi, gunicorn, and nginx (refer to https://github.com/shreyakarthik1210/WebStreamingSecurityCamera)
2. Create a virtual environment with python version 3.9.3. Commands:
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz
tar -xf Python-3.7.9.tar.xz
cd Python-3.7.9
./configure
make -j4  # Adjust the number of cores based on your Raspberry Pi model
sudo make altinstall
3. Clone github repo into virtual environment and run code.
