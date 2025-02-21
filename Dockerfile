# Official Python image (slim version)
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install requirements.txt
# RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Default Command
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8080"]

# open http://localhost:8080