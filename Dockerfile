FROM pytorch/pytorch:latest

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY lib training predict.py classifier.pt ./

# ENTRYPOINT [ "python", "predict.py" ]