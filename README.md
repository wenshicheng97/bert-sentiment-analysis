## Set up environment

Set up a conda environment for packages used in this project.
```
conda create -n bert-sa python=3.8
conda activate bert-sa
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk
```
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
>>> nltk.download('stopwords')
>>> nltk.download('words')
>>> exit()
```