FROM python:3.8.6-buster
COPY . /.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD streamlit run streamlit_setup/app.py --server.port $PORT