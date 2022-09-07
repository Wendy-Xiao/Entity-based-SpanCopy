import spacy
from datasets import load_dataset
import torch
import pdb
import sys
from transformers import PegasusTokenizerFast
import os


def get_entities(nlp, doc):
    s = nlp(doc)
    all_entities = [ent.text for ent in s.ents]
    all_entities_pos = [(ent.start_char, ent.end_char) for ent in s.ents]
    return all_entities, all_entities_pos


def build_dataset(dataset_name, orig_dataset):
    nlp = spacy.load("en_core_web_sm")
    new_dataset = []
    for i, d in enumerate(orig_dataset):
        if dataset_name == "cnndm":
            summary = orig_dataset[i]["highlights"]
            document = orig_dataset[i]["article"].replace("(CNN)", "")
            d["document"] = document
        summ_entities, summ_ent_pos = get_entities(nlp, summary)
        doc_entities, doc_ent_pos = get_entities(nlp, document)
        d["document"] = document
        d["summary"] = summary
        d["summ_entities"] = summ_entities
        d["summ_ent_pos"] = summ_ent_pos
        d["doc_entities"] = doc_entities
        d["doc_ent_pos"] = doc_ent_pos
        new_dataset.append(d)
        if i % 1000 == 0:
            print("%d finished" % (i))
            sys.stdout.flush()
    return new_dataset


if __name__ == "__main__":
    dataset_name = "cnndm"
    orig_data = load_dataset(
        "cnn_dailymail", "2.0.0", cache_dir="/scratch/wenxiao/topic_model/dataset/"
    )
    all_data = dict()
    for d in ["train", "validation", "test"]:
        all_data[d] = build_dataset(dataset_name, orig_data[d])
    torch.save(all_data, "/scratch/wenxiao/topic_model/dataset/cnndm_with_ent.pt")

