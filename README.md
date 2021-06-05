# Entity Enriched Neural Models for Clinical Question Answering

[Paper](https://www.aclweb.org/anthology/2020.bionlp-1.12/) | [Leaderboard](https://emrqa.github.io/) | [emrQA repo](https://github.com/panushri25/emrQA) |  

This repo contains the code for the paper: Entity Enriched Neural Models for Clinical Question Answering [pdf](https://www.aclweb.org/anthology/2020.bionlp-1.12/). The paper was published at BioNLP workshop at ACL'20. 

Abstract
>We explore state-of-the-art neural models for question answering on electronic medical records and improve their ability to generalize better on previously unseen (paraphrased) questions at test time. We enable this by learning to predict logical forms as an auxiliary task along with the main task of answer span detection. The predicted logical forms also serve as a rationale for the answer. Further, we also incorporate medical entity information in these models via the ERNIE architecture. We train our models on the large-scale emrQA dataset and observe that our multi-task entity-enriched models generalize to paraphrased questions ~5% better than the baseline BERT model.

![Figure](/imgs/cERNIE.png)

# Requirements

- All requirements are provided in requirements.txt.
- pip or conda both works for creating the environment. 


# Generate para-level emrQA dataset. 

- The emrQA data can be extracted using the emrQA [GitHub repo.](https://github.com/panushri25/emrQA) or by registering on n2c2 and downloading the data from under the Community Annotations Downloads tab (Question answering dataset generated from 2014 Heart disease risk factors data) here: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ (contrary to what it says on the download link, the question answering dataset has been generated from all of the previous i2b2 challenges)
- We recommend creating the data with the help of the ipynbs provided in the folder ```/ipynbs```.
    - The ipynb 'get_emrqa_para_level_data.ipynb' would require these 6 files: medication-qa.json, relations-qa.json, risk-qa.json, risk-ql.csv, relations-ql.csv, medication-ql.csv which can be extracted from emrQA [GitHub repo](https://github.com/panushri25/emrQA).
    - Run the ipynb 'get_emrqa_para_level_data.ipynb' to get the emrQA para-level data. A snapshot of the data is shown below.
    ![emrqa](/imgs/emrqa.png)
    - To get the entity level information using Metamap, run the ipynb 'get_entity_information.ipynb'
        - We used ```pymetamap``` [package](https://github.com/AnthonyMRios/pymetamap). This is quite slow and you can also use [scispacy](https://github.com/allenai/scispacy) to extract this information but it is a bit less accurate than using the MetaMap itself. A snapshot of the data is shown below. 
    ![emrqa](/imgs/emrqa_entity.png)
    - The scripts for both ipynbs are also provided in the folder ```/scripts```.
    
- The generated dataset consists of both:
    - `strict split`: emrQA para-level data split according to the question templates.
    - `normal split`: emrQA para-level data split randomly.
    ```
     data
       ├──'split'
       │     ├── 'train'
       │     ├── 'dev'
       │     └── 'test'
       └──'strict_split'
             ├── 'train'
             ├── 'dev'
             └── 'test'
    ```
# ERNIE

- For ERNIE, `--model_name_or_path` should also have `ernie_config.json` inside the path. It is same as the usual bert_config.json with an additional line which provides the layer types of the ERNIE model. The required line is:
    - ```layer_types": ["sim", "sim", "sim", "sim", "sim", "mix", "norm", "norm", "norm", "norm", "norm", "norm"]``` 

# Training

- Training a multi-task clinical BERT model | Strict setting 
```python train.py --data_dir=../data/emrqa_parawise_data_w_entities.pkl --model_name_or_path=<BERT_path> --model_type=bert model_save_name=cBERT --train_setting=strict --do_train --do_eval --do_test --train_batch_size=6 --train_epochs=3 --lr=2e-5 --warmup_proportion=0.1 --auxiliary_task_wt=0.4 --gpu=1```


- Training a multi-task clinical ERNIE model | Strict setting
```python train.py --data_dir=../data/emrqa_parawise_data_w_entities.pkl --model_name_or_path=<BERT_path> --model_type=ernie model_save_name=cERNIE --train_setting=strict --do_train --do_eval --do_test --train_batch_size=6 --train_epochs=3 --lr=2e-5 --warmup_proportion=0.1 --auxiliary_task_wt=0.4 --gpu=1```


- Training a multi-task clinical ERNIE model | Strict setting
```python train.py --data_dir=../data/emrqa_parawise_data_w_entities.pkl --model_name_or_path=<BASE_MODEL_PATH> --model_type=ernie --multi_task model_save_name=mCERNIE --train_setting=strict --do_train --do_eval --do_test --train_batch_size=6 --train_epochs=3 --lr=2e-5 --warmup_proportion=0.1 --auxiliary_task_wt=0.4 --gpu=1```

  


# Paper
If you use our benchmark or the code in this repo, please cite our paper `\cite{rawat-etal-2020-entity}` or `\cite{rawat2020entity}`.
```
@inproceedings{rawat-etal-2020-entity,
    title = "Entity-Enriched Neural Models for Clinical Question Answering",
    author = "Rawat, Bhanu Pratap Singh  and
      Weng, Wei-Hung  and
      Min, So Yeon  and
      Raghavan, Preethi  and
      Szolovits, Peter",
    booktitle = "Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bionlp-1.12",
    doi = "10.18653/v1/2020.bionlp-1.12",
    pages = "112--122",
}

@article{rawat2020entity,
  title={Entity-Enriched Neural Models for Clinical Question Answering},
  author={Rawat, Bhanu Pratap Singh and Weng, Wei-Hung and Min, So Yeon and Raghavan, Preethi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2005.06587},
  year={2020}
}
```
   
