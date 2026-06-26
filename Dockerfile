# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run your Streamlit application
CMD ["streamlit", "run", "deployment/frontend/app_ui.py", "--server.address=0.0.0.0"]