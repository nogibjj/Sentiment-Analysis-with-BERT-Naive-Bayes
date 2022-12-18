[![CI](https://github.com/nogibjj/Sentiment-Analysis-with-BERT-Naive-Bayes/actions/workflows/cicd.yml/badge.svg)](https://github.com/nogibjj/Sentiment-Analysis-with-BERT-Naive-Bayes/actions/workflows/cicd.yml)
[![Codespaces Prebuilds](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds)

# Sentiment Analysis on IMDb data with Naive Bayes and BERT Models

## Folder Structure
**Project Code** <br>
|-- data_cleaning.py <br>
|-- naive_bayes_model.py <br>
|-- bert_model.py <br>
|-- bert_predict.py <br>
|-- requirements.txt <br>
**README.md** <br>
**NLP_Final_Project.pdf** <br>




### Verify GPU works

The following examples test out the GPU

* run pytorch training test: `python utils/quickstart_pytorch.py`
* run pytorch CUDA test: `python utils/verify_cuda_pytorch.py`
* run tensorflow training test: `python utils/quickstart_tf2.py`
* run nvidia monitoring test: `nvidia-smi -l 1` it should show a GPU
* run whisper transcribe test `./utils/transcribe-whisper.sh` and verify GPU is working with `nvidia-smi -l 1`

