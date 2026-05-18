# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file first to take advantage of Docker caching
COPY requirements.txt /code/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application files into the container
COPY . /code

# Change permissions to allow writing within the app folder (required by Hugging Face)
RUN chmod -R 777 /code

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Start the FastAPI app on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
