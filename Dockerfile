# Use the official PyTorch container as the base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install Git
RUN apt-get update && apt-get install -y git

# Install Chromium and ChromeDriver without prompting for user input
RUN apt-get install -y chromium-chromedriver
RUN apt-get install -y chromium-browser

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /app
COPY . /workspace

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt