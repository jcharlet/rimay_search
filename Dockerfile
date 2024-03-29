FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN \
  apt-get install -y curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
  apt-get install git-lfs

RUN git clone https://github.com/jcharlet/rimay_search.git .

RUN pip3 install -r requirements.txt

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/visualization/visualize.py", "--server.port=7860", "--server.address=0.0.0.0"]