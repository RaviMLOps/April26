# pull python base image
FROM python:3.11-bullseye

# copy application files
ADD /. /model/

# specify working directory
WORKDIR /model

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app.py"]