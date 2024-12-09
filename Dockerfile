FROM public.ecr.aws/docker/library/python:3.12-slim

# Install AWS Lambda Web Adapter
# Adapter allows you to use AWS Lambda as a HTTP service
# to avoid using mangum service
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set environment variables
ENV PORT=8080

WORKDIR /var/task

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir  -r requirements.txt

# Copy application code and model
COPY app.py ./
COPY bird_200_classes.txt ./
COPY food_101_classes.txt ./
#COPY traced_models/ ./traced_models/
COPY onxx_models/ ./onxx_models/
COPY sample_data/ ./sample_data/
EXPOSE 8080
# Set command
CMD exec uvicorn --host 0.0.0.0 --port $PORT app:app 