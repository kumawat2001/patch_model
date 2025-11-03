FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy your script and files
COPY patch_model.py .

COPY thumbnail.png .
COPY Requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r Requirement.txt

# Default command to run your prediction script
CMD ["python", "patch_model.py"]

