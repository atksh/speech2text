FROM atksh/python-gpu:48997bab89fa

WORKDIR /code
COPY requirements.txt /code/
RUN python -m pip install -r requirements.txt \
    && pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
COPY . .

ENTRYPOINT /bin/bash