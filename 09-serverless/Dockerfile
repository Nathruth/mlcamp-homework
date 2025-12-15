# Dockerfile
FROM agrigorev/model-2025-hairstyle:v1

WORKDIR /var/task

# Copy your lambda handler code
COPY lambda_handler.py .

# Install required Python packages
RUN pip install --no-cache-dir pillow numpy requests onnxruntime
