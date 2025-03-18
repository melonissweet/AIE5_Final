FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install build dependencies
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Add user - this is the user that will run the app
RUN useradd -m -u 1000 user

# Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH        

ENV UVICORN_WS_PROTOCOL=websockets

# Set the working directory
WORKDIR $HOME/app

# Copy the app to the container
COPY . $HOME/app

# Set proper permissions
RUN chown -R user:user $HOME/app

# Switch to user for package installation
USER user

# Install dependencies with uv
RUN uv sync

# Update PATH to include uv-installed executables
ENV PATH=$HOME/app/.venv/bin:$PATH

# Expose ports
EXPOSE 8000 8501

# Create a properly formatted run script (no need to start Redis here)
USER root
RUN echo '#!/bin/bash' > $HOME/app/run.sh && \
    echo '# Start the application' >> $HOME/app/run.sh && \
    echo 'cd $HOME/app' >> $HOME/app/run.sh && \
    echo 'uvicorn main:app --host 0.0.0.0 --port 8000 &' >> $HOME/app/run.sh && \
    echo 'cd $HOME/app && streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0' >> $HOME/app/run.sh && \
    chmod +x $HOME/app/run.sh

# Command to run the application
CMD ["/bin/bash", "/home/user/app/run.sh"]