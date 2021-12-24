FROM python:3.9.6-slim
COPY test /test
COPY requirements.txt /requirements.txt
COPY jupyter_notebook_config.py /jupyter_notebook_config.py
RUN pip install -r requirements.txt
RUN chmod 400 /test/Task0/test_A.ipynb
WORKDIR /test
CMD ["jupyter-notebook","--config","/jupyter_notebook_config.py", "--debug"]
