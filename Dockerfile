
# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /mle-training-

# Copy the requirements.txt file
COPY ./requirements.txt .

# update pip
RUN pip install --upgrade pip

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /mle-training
COPY . .

# Make port 8080 available to the world outside this container
EXPOSE 8080

CMD ["python", "main.py"]

