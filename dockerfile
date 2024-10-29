FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir gradio opencv-python-headless numpy
EXPOSE 7860
CMD ["python", "app.py"]
