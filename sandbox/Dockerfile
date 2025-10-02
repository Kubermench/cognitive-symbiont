FROM python:3.11-slim

WORKDIR /srv/app

# Base requirements for Symbiont core
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Service-specific (sandbox) requirements if present
COPY sandbox/requirements.txt /tmp/sandbox-requirements.txt
RUN pip install --no-cache-dir -r /tmp/sandbox-requirements.txt

# Copy full project and install in editable mode for CLI parity
COPY . /srv/app
ENV PYTHONPATH=/srv/app

EXPOSE 8000

CMD ["uvicorn","sandbox.app.main:app","--host","0.0.0.0","--port","8000"]
