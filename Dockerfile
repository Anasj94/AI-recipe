# Use an official Python runtime as a parent image
FROM python:3.12.1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install HDF5 library and development files
RUN apt-get update && \
    apt-get install -y libhdf5-dev

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container (if your app listens on a specific port)
EXPOSE 80

# Define environment variable (if needed)
ENV NAME World

# Command to run the Python application
CMD ["streamlit", "run", "app.py"]
