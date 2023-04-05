import os
import subprocess

# Set project directory
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# Set ByteTrack directory
byte_track_dir = os.path.join(script_dir, 'ByteTrack')

# Install requirements
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)

# Clone ByteTrack repository
subprocess.run(['git', 'clone', 'https://github.com/ifzhang/ByteTrack.git'], check=True)

# Update the version of ONNX that is used by ByteTrack
with open('./ByteTrack/requirements.txt', 'r') as file:
    byte_track_requirements = file.read()

new_byte_track_requirements = byte_track_requirements.replace('onnx==1.8.1', 'onnx==1.9.0')

with open('./ByteTrack/requirements.txt', 'w') as file:
    file.write(new_byte_track_requirements)
    
# Install ByteTrack requirements
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], cwd=byte_track_dir, check=True)

# Set up ByteTrack
subprocess.run(['python', 'setup.py', 'develop'], cwd=byte_track_dir, check=True)