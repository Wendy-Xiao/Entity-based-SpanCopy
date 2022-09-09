import spacy
from datasets import load_dataset
import torch
import pdb
import sys
from spacy.lang.en.stop_words import STOP_WORDS
import re
import json


def approximate_match_number(entity_list1, entity_list2):
    match_num = 0
    for e1 in entity_list1:
        for e2 in entity_list2:
            if (
                e1.lower() == e2.lower()
                or e1.replace("the", "").strip() == e2
                or e1 == e2.replace("the", "").strip()
            ):
                match_num += 1
                break
    return match_num


def get_entities(nlp, doc):
    s = nlp(doc)
    all_entities = [ent.text for ent in s.ents]
    all_entities_pos = [(ent.start_char, ent.end_char) for ent in s.ents]
    return all_entities, all_entities_pos


def get_noun_chunks(nlp, doc):
    s = nlp(doc)
    # remove stop words
    noun_chunks = [n for n in s.noun_chunks if n.text.lower() not in STOP_WORDS]
    all_noun_chunks = [n.text.strip() for n in noun_chunks]
    all_noun_chunks_pos = [(n.start_char, n.end_char) for n in noun_chunks]
    return all_noun_chunks, all_noun_chunks_pos


def preprocessing_arxiv_orig(single_data, length_limit=4096):
    document = [" ".join(s) for s in single_data["sections"]]
    summary = " ".join(
        [
            sent.replace("<S>", "").replace("</S>", "").strip()
            for sent in single_data["abstract_text"]
        ]
    )
    l_per_doc = length_limit // len(document)
    document = [" ".join(d.split(" ")[:l_per_doc]) for d in document]
    document = " <doc-sep> ".join(document)
    return document, summary


def build_dataset(
    dataset_name,
    orig_dataset,
    t="entity",
):
    nlp = spacy.load("en_core_web_sm")
    new_dataset = []
    for i, d in enumerate(orig_dataset):
        if dataset_name == "cnndm":
            summary = orig_dataset[i]["highlights"]
            document = orig_dataset[i]["article"].replace("(CNN)", "")
        elif dataset_name == "pubmed" or dataset_name == "arxiv":
            summary = orig_dataset[i]["abstract"]
            document = orig_dataset[i]["article"]
        elif dataset_name == "xsum":
            summary = orig_dataset[i]["summary"]
            document = orig_dataset[i]["document"]
        elif dataset_name == "multi_news":
            summary = orig_dataset[i]["summary"].replace("–", "").strip()
            document = orig_dataset[i]["document"].replace("|||||", "\n")
            document = re.sub(r"\n\s+", "\n ", document)
            if len(document) > 90000:
                print("%d is too long" % (i))
                sys.stdout.flush()
                continue
        elif dataset_name == "arxiv_primera":
            document, summary = preprocessing_arxiv_orig(orig_dataset[i])
        if len(document) == 0 or len(summary) == 0:
            print("%d is empty" % (i))
            sys.stdout.flush()
            continue
        if t == "entity":
            summ_entities, summ_ent_pos = get_entities(nlp, summary)
            doc_entities, doc_ent_pos = get_entities(nlp, document)
        elif t == "noun_chunks":
            summ_entities, summ_ent_pos = get_noun_chunks(nlp, summary)
            doc_entities, doc_ent_pos = get_noun_chunks(nlp, document)
        new_data = {}
        new_data["document"] = document
        new_data["summary"] = summary
        new_data["summ_entities"] = summ_entities
        new_data["summ_ent_pos"] = summ_ent_pos
        new_data["doc_entities"] = doc_entities
        new_data["doc_ent_pos"] = doc_ent_pos
        new_dataset.append(new_data)
        if i % 1000 == 0:
            print("%d finished" % (i))
            sys.stdout.flush()
    return new_dataset


def get_filtered_dataset(orig_dataset):
    new_dataset = []
    for i, d in enumerate(orig_dataset):
        summ_entities = set(d["summ_entities"])
        doc_entities = set(d["doc_entities"])
        match_num = approximate_match_number(summ_entities, doc_entities)
        if match_num == len(summ_entities) and len(summ_entities) != 0:
            new_dataset.append(d)
    return new_dataset


if __name__ == "__main__":
    # cnndm
    # dataset_name = "cnndm"
    # orig_data = load_dataset(
    #     "cnn_dailymail", "2.0.0", cache_dir="./dataset/"
    # )
    # output_dir = "./dataset/cnndm_with_nounchunk/"
    # updated_dir = "./dataset/cnndm_with_nounchunk_filtered/"
    # for d in ["test", "validation", "train"]:
    #     all_data = build_dataset(dataset_name, orig_data[d], t="noun_chunks")
    #     torch.save(all_data, output_dir + "%s.pt" % (d))
    #     print("number of data in %s: %d" % (d, len(all_data)))
    #     all_data = get_filtered_dataset(all_data)
    #     torch.save(all_data, updated_dir + "%s.pt" % (d))
    #     print("number of filtered data in %s: %d" % (d, len(all_data)))

    # pubmed
    # dataset_name = "arxiv"
    # orig_data = load_dataset(
    #     "scientific_papers",
    #     dataset_name,
    #     cache_dir="./dataset/",
    # )
    # output_dir = "./dataset/%s_with_nounchunk/" % (
    #     dataset_name
    # )
    # updated_dir = "./dataset/%s_with_nounchunk_filtered/" % (
    #     dataset_name
    # )
    # for d in ["test", "validation", "train"]:
    #     all_data = build_dataset(dataset_name, orig_data[d], t="noun_chunks")
    #     torch.save(all_data, output_dir + "%s.pt" % (d))
    #     print("number of data: %d" % (len(all_data)))

    #     all_data = get_filtered_dataset(all_data)
    #     torch.save(all_data, updated_dir + "%s.pt" % (d))
    #     print("number of data: %d" % (len(all_data)))

    # xsum
    # dataset_name = "multi_news"
    # orig_data = load_dataset(
    #     dataset_name, cache_dir="./dataset/"
    # )
    # output_dir = "./dataset/%s_with_nounchunk/" % (
    #     dataset_name
    # )
    # updated_dir = "./dataset/%s_with_nounchunk_filtered/" % (
    #     dataset_name
    # )
    # # all_data = dict()
    # for d in ["train"]:
    #     # orig_data = torch.load(orig_dir + "%s.pt" % (d))
    #     all_data = build_dataset(dataset_name, orig_data[d], t="noun_chunks")
    #     torch.save(all_data, output_dir + "%s.pt" % (d))
    #     print("number of data in %s: %d" % (d, len(all_data)))
    #     all_data = get_filtered_dataset(all_data)
    #     torch.save(all_data, updated_dir + "%s.pt" % (d))
    #     print("number of data: %d" % (len(all_data)))

    # multi-news
    # dataset_name = "multi_news"
    # orig_data = load_dataset(
    #     "multi_news", cache_dir="/scratch/wenxiao/topic_model/dataset/"
    # )
    # output_dir = "/scratch/wenxiao/topic_model/dataset/%s_with_ent/" % (dataset_name)
    # updated_dir = "/scratch/wenxiao/topic_model/dataset/%s_with_ent_filtered/" % (
    #     dataset_name
    # )
    # for d in ["test", "validation", "train"]:
    # for d in ["test"]:
    #     all_data = build_dataset(dataset_name, orig_data[d])
    #     torch.save(all_data, output_dir + "%s.pt" % (d))
    # orig_data = torch.load(output_dir + "%s.pt" % (d))
    # all_data = get_filtered_dataset(orig_data)
    # torch.save(all_data, updated_dir + "%s.pt" % (d))
    # print("number of data: %d" % (len(all_data)))

    # arxiv_orig
    dataset_name = "arxiv_primera"
    orig_data = {}
    with open("./dataset/arxiv-dataset/train.txt", "r") as of:
        all_lines = of.readlines()
    orig_data["train"] = [json.loads(l) for l in all_lines]
    with open("./dataset/arxiv-dataset/val.txt", "r") as of:
        all_lines = of.readlines()
    orig_data["validation"] = [json.loads(l) for l in all_lines]

    with open("./dataset/arxiv-dataset/test.txt", "r") as of:
        all_lines = of.readlines()
    orig_data["test"] = [json.loads(l) for l in all_lines]

    output_dir = "./dataset/%s_with_ent/" % (dataset_name)
    updated_dir = "./dataset/%s_with_ent_filtered/" % (dataset_name)
    # all_data = dict()
    for d in ["test", "validation", "train"]:
        # orig_data = torch.load(orig_dir + "%s.pt" % (d))
        all_data = build_dataset(dataset_name, orig_data[d])
        torch.save(all_data, output_dir + "%s.pt" % (d))
        print("number of data in %s: %d" % (d, len(all_data)))
        all_data = get_filtered_dataset(all_data)
        torch.save(all_data, updated_dir + "%s.pt" % (d))
        print("number of data: %d" % (len(all_data)))
