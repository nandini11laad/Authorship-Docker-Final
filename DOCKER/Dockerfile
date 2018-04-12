FROM python:2.7.12

# Set the working directory to /home/lora/Desktop/DOCKER
WORKDIR /home/lora/Desktop/DOCKER/

# Copy the current directory contents into the container at /home/lora/Desktop/DOCKER
ADD . /home/lora/Desktop/DOCKER/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 5008


# Run app.py when the container launches
CMD ["python", "projfinal.py"]
