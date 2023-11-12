FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install flake8 pytest torch torchvision
CMD ["python", "-m", "pytest", "-v", "./test/test_resnet.py"]
