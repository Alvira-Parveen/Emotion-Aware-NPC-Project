FROM python:3.11-slim

# Install system dependencies (ffmpeg is for moviepy, fonts for PIL text drawing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# (Hugging Face Spaces require running as a non-root user)
RUN useradd -m -u 1000 user
USER user

# Set home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy application files with proper permissions
COPY --chown=user . $HOME/app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# IMPORTANT: Pre-download the heavy MTCNN weights into the Docker image!
# This ensures that when your app starts, it doesn't need to download anything.
RUN python -c "from fer import FER; print('Downloading weights...'); FER(mtcnn=True); print('Done!')"

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

# Start Gunicorn on port 7860 with a generous timeout
CMD ["gunicorn", "-b", "0.0.0.0:7860", "web_app:app", "--timeout", "120"]
