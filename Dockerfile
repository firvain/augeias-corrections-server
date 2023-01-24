FROM python:3.9
RUN mkdir /app
WORKDIR /app
COPY ./requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD [ "python", "./main.py" ]
