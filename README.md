# Entity-based-SpanCopy
Official code for [Entity-based SpanCopy for Abstractive Summarization to Improve the Factual Consistency](https://arxiv.org/abs/2209.03479)

## Dependencies
python == 3.7.11

pytorch == 1.10.0

pytorch_lightning == 1.5.3

transformers == 4.11.3

## Usage
### SpanCopy Model
To train the SpanCopy model with global relevance, run with `--use_global_relevance` with proper `--beta`. The beta used in our experiments is set by grid search on small subsets  of each dataset (2k for training and 200 for validation). 

| Dataset | beta |
| --- | ----------- |
| CNNDM | 0.5| 
| CNNDM-filtered | 0.5 |
| XSum | 0.6 | 
| XSum-filtered | 0.9 |
| Pubmed | 0.4 | 
| Pubmed-filtered | 0.4 |
| arXiv | 0.4 | 
| arXiv-filtered | 0.5 |

To train the SpanCopy model, run without above two options.
```
python spanCopyTrainer.py --gpus 1 \
                                --batch_size 4 \
                                --label_smoothing 0.1 \
                                --model_path /path/to/where/you/want/to/save/the/model \
                                --data_path /path/to/the/data/folder \
                                --model_name pegasus-cnndm \
                                --dataset_name cnndm \
                                --adafactor \
                                --use_global_relevance \
                                --beta 0.5 \          
```

To test the model
```
python spanCopyTrainer.py --mode test \
                                --resume_ckpt /path/to/the/checkpoint \
                                --gpus 1 \
                                --batch_size 4 \
                                --label_smoothing 0.1 \
                                --model_path /path/to/the/model/folder/where/you/want/to/save/summaries \
                                --data_path /path/to/the/data/folder \
                                --model_name pegasus-cnndm \
                                --dataset_name cnndm \
                                --use_global_relevance \
                                --beta 0.5 \
                                --save_gr
``` 
### Pegasus bsl
To train/test the Pegasus model, simply run the `pegasus_trainer.py` with the same settings specified above.

## Datasets
All the filtered/unfiltered datasets (CNNDM/Xsum/Pubmed/arXiv) can be found [here](https://drive.google.com/drive/folders/1I_NhJ44VVnaZ6GWY0hNjoLdYoCVgd_KV?usp=sharing).

