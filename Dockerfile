# Dockerfile

# Start with a standard Python base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# --- FIX: COPY THE CERTIFICATE FIRST ---
# Copy the custom corporate certificate into the container
COPY trusted_certs.crt /usr/local/share/ca-certificates/trusted_certs.crt

# --- FIX: UPDATE THE CONTAINER'S CERTIFICATE STORE ---
# Run the command to make the OS trust our custom certificate
RUN update-ca-certificates

# Copy the requirements file
COPY requirements.txt .

# --- FIX: CONFIGURE PIP TO USE THE CERTIFICATE ---
# Tell pip to use the updated trusted certificate bundle for all subsequent network calls
RUN pip config set global.cert /etc/ssl/certs/ca-certificates.crt && \
    pip install --no-cache-dir -r requirements.txt

# Copy all your application code and utilities
# (No need to copy the cert again, it's already in the trusted store)
COPY mcp_app.py .
COPY retriever_azure.py .
COPY utils/ ./utils/

# Copy the offline models into the image
#COPY blip-model-offline/ ./blip-model-offline/
#COPY bge-model-offline/ ./bge-model-offline/

# Expose the port your MCP server runs on
EXPOSE 9002

# The command to run your application when the container starts
CMD ["python", "mcp_app.py"]