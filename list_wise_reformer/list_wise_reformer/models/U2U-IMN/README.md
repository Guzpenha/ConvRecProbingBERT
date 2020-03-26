# Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots
This repository contains the source code and datasets for the IEEE/ACM Transactions on Audio, Speech and Language Processing paper [Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://ieeexplore.ieee.org/document/8910440) by Gu et al. <br>

## Dependencies
Python 2.7 <br>
Tensorflow 1.4.0

## Datasets
Your can download the processed datasets used in our paper here and unzip it to the folder of ```data```. <br>
[Ubuntu_V2](https://drive.google.com/open?id=1tS_VC47z8CVPr-tZu0U4JEEwBT04N6ks) <br>

## Train a new model
```
cd scripts
bash ubuntu_train.sh
```
The training process is recorded in ```log_train_UbuntuV2.txt``` file.

## Test a trained model
```
bash ubuntu_test.sh
```
The testing process is recorded in ```log_test_UbuntuV2.txt``` file. And your can get a ```ubuntu_test_out.txt``` file which records scores for each context-response pair. Run the following command your can compute the metric of Recall.
```
python compute_recall.py
```

## Cite
If you use the code and datasets, please cite the following paper:
**"Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots"**
Jia-Chen Gu, Zhen-Hua Ling, Quan Liu. _TASLP_

```
@ARTICLE{8910440,
         author={J. {Gu} and Z. {Ling} and Q. {Liu}},
         journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
         title={Utterance-to-Utterance Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots},
         year={2020},
         volume={28},
         number={},
         pages={369-379},
         keywords={Context modeling;Encoding;Buildings;Neural networks;Training;Dialogue;response selection;interactive matching network;utterance-to-utterance},
         doi={10.1109/TASLP.2019.2955290},
         ISSN={2329-9304},
         month={},}
}
```
