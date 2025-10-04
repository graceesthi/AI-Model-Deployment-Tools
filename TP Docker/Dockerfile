# Use Ubuntu base image
FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# Set working directory
WORKDIR /app/yolov5

# Install Python dependencies from YOLOv5 requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directories for input and output
RUN mkdir -p /app/myimages/input /app/myimages/output

# Create a script to run the detector
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
sys.path.append("/app/yolov5")\n\
import torch\n\
from detect import run\n\
\n\
# Load model and run detection\n\
run(weights="yolov5s.pt",\n\
    source="/app/myimages/input",\n\
    project="/app/myimages/output",\n\
    exist_ok=True)\n\
\n\
print("Detection completed. Results saved in output directory.")\n\
' > /app/run_detection.py

# Make the script executable
RUN chmod +x /app/run_detection.py

# Set the entry point
ENTRYPOINT ["python3", "/app/run_detection.py"]