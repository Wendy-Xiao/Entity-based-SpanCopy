from os import truncate
from pandas import cut
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.feature_extraction.text import CountVectorizer
import pdb
import numpy as np
from transformers import PegasusTokenizerFast
import spacy


def get_entities(nlp, doc):
    s = nlp(doc)
    all_entities = [ent.text for ent in s.ents]
    all_entities_pos = [(ent.start_char, ent.end_char) for ent in s.ents]
    return all_entities, all_entities_pos


class SummDataset(Dataset):
    def __init__(
        self, hf_dataset, dataset_name, tokenizer, max_length_input, max_length_output
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        # if self.dataset_name == "cnndm":
        #     document = entry["article"]
        #     summary = entry["highlights"]
        # elif (
        #     self.dataset_name == "pubmed"
        #     or self.dataset_name == "arxiv"
        #     or self.dataset_name == "xsum"
        # ):
        document = entry["document"]
        summary = entry["summary"]
        # if "pegasus" in self.tokenizer.__class__.__name__.lower():
        #     document = document.replace("\n", "<n>")
        #     summary = summary.replace("\n", "<n>")
        input_ids = self.tokenizer(
            document, truncation=True, max_length=self.max_length_input
        )
        output_ids = self.tokenizer(
            summary, truncation=True, max_length=self.max_length_output
        )
        return (
            torch.tensor(input_ids["input_ids"]),
            torch.tensor(output_ids["input_ids"]),
        )


class SummTMJointDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, vectorizer):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        document = entry["article"]
        summary = entry["highlights"]
        input_ids = self.tokenizer.encode(
            self.tokenizer.tokenize(document), truncatiton=True
        )
        output_ids = self.tokenizer.encode(
            self.tokenizer.tokenize(summary), truncatiton=True
        )
        s_bow = self.vectorizer.transform([summary]).toarray()
        s_bow = s_bow.astype(np.float32)
        return input_ids, output_ids, s_bow[0]


class SummEntitiesDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        document = entry["article"]
        summary = entry["highlights"]
        entities = entry["entities"]
        input_ids = self.tokenizer.encode(
            document, truncation=True, max_length=self.max_length
        )
        output_ids = self.tokenizer.encode(
            summary, truncation=True, max_length=self.max_length
        )
        summ_entities = self.tokenizer.encode(" ".join(entities), truncation=False)
        summ_entities_vec = torch.zeros(self.tokenizer.vocab_size)
        for wid in summ_entities:
            if (
                wid != self.tokenizer.unk_token_id
                and wid != self.tokenizer.eos_token_id
            ):
                summ_entities_vec[wid] += 1
        return torch.tensor(input_ids), torch.tensor(output_ids), summ_entities_vec


class spanCopySummDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        dataset_name,
        tokenizer,
        max_length_input,
        max_length_output,
        contains_entity=True,
        debug=False,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output
        self.dataset_name = dataset_name
        self.contains_entity = contains_entity
        if not self.contains_entity:
            self.spacy_model = spacy.load("en_core_web_sm")
        self.debug = debug

    def __len__(self):
        return len(self.hf_dataset)

    def map_entity_input_output(self, summ_entities, doc_entities):
        entity_match = {}
        for i_e1, e1 in enumerate(summ_entities):
            for i_e2, e2 in enumerate(doc_entities):
                if (
                    e1.lower() == e2.lower()
                    or e1.replace("the", "").strip() == e2
                    or e1 == e2.replace("the", "").strip()
                ):
                    entity_match[i_e1] = i_e2
                    break
        return entity_match

    def cut_entity_list(self, entity_list, ent_pos, max_char):
        ind = 0
        while ind < len(ent_pos) and ent_pos[ind][1] < max_char:
            ind += 1
        return entity_list[:ind], ent_pos[:ind]

    def map_entities_to_tokens(self, input_ids, char_token_mapping, doc_ent_pos):
        char_token_mapping = char_token_mapping[:-1]  # remove (0,0) in the end
        mapping = torch.zeros(len(input_ids), len(doc_ent_pos))
        cur_token = 0
        for i, ent in enumerate(doc_ent_pos):

            while (
                cur_token < len(char_token_mapping)
                and char_token_mapping[cur_token][0] < ent[0]
            ):
                cur_token += 1
            # if reach the end, make the last token match the beginning of current entity
            if cur_token == len(char_token_mapping) or (
                cur_token > 0 and char_token_mapping[cur_token][0] > ent[0]
            ):
                start = cur_token - 1
            else:
                start = cur_token
            # if reach the end, make the last token match the end of current entity
            while (
                cur_token < len(char_token_mapping)
                and char_token_mapping[cur_token][1] < ent[1]
            ):
                cur_token += 1
            end = cur_token
            # if end - start == 0:
            #     end += 1
            mapping[start : (end + 1), i] = 1 / (end + 1 - start)
        # pdb.set_trace()
        return mapping

    def map_orig_output_to_entity_output_and_build_labels(
        self, output_ids, char_token_mapping, summ_ent_pos, entity_match
    ):
        # output_ids: id list for summaries
        # char_token_mapping:   the start and end indices of each token w.r.t. characters
        #                       list of tuples, [(start_1,end_1),(start_2,end_2)], where start and end
        #                       are char-based indices
        # summ_ent_pos:     the start and end indices of each entities w.r.t charactors
        #                   list of tuples, [(start_1,end_1),(start_2,end_2)], where start and end
        #                       are char-based indices
        # entity_match:     the mapping between the document entities and summary entities
        #                   {e_1^s:e_1^d,e_2^s:e_2^d}, each key is an index of an entity in the summary
        #                   the corresponding value is the index of the entity in the document.
        # Return:   labels: the actual labels to train on range between [0,|V|+|E|], where |E| is the number of
        #                   entities in the original document. len(labels)<=len(output_ids)
        #           mapping: the mapping between output_ids to labels, with shape (output_ids,labels),
        #
        mapping = []
        cur_token = 0
        labels = []
        for i_ent, ent in enumerate(entity_match.keys()):
            # There might be the case where the token indices between
            # char_token_mapping and summ_ent_pos has difference 1
            while (
                cur_token < len(char_token_mapping) - 2
                and char_token_mapping[cur_token][0] < summ_ent_pos[ent][0]
            ):
                cur_mapping = torch.zeros(len(output_ids))
                cur_mapping[cur_token] = 1
                mapping.append(cur_mapping)
                labels.append(output_ids[cur_token])
                cur_token += 1
            if (
                cur_token > 0
                and cur_token < len(char_token_mapping) - 2
                and char_token_mapping[cur_token][0] > summ_ent_pos[ent][1]
            ):
                start = cur_token - 1
                mapping.pop(-1)
                labels.pop(-1)
            else:
                start = cur_token
            while (
                cur_token < len(char_token_mapping) - 1
                and char_token_mapping[cur_token][1] < summ_ent_pos[ent][1]
            ):
                cur_token += 1

            end = cur_token
            cur_mapping = torch.zeros(len(output_ids))
            cur_mapping[start : (end + 1)] = 1 / (end + 1 - start)
            cur_token += 1
            mapping.append(cur_mapping)
            labels.append(self.tokenizer.vocab_size + entity_match[ent])
        while cur_token < len(output_ids):
            cur_mapping = torch.zeros(len(output_ids))
            cur_mapping[cur_token] = 1
            mapping.append(cur_mapping)
            labels.append(output_ids[cur_token])
            cur_token += 1
        # pdb.set_trace()
        assert len(labels) == len(mapping)
        assert len(labels) <= len(output_ids)

        mapping = torch.stack(mapping, dim=1)
        # assert (mapping.sum(0) == 1).all()
        return mapping, labels

    def process_single_data(self, entry):

        if self.dataset_name == "cnndm":
            document = entry["document"].replace("(CNN)", "")
            summary = entry["summary"]
        elif (
            self.dataset_name == "pubmed"
            or self.dataset_name == "arxiv"
            or self.dataset_name == "xsum"
            or self.dataset_name == "multi_news"
        ):
            document = entry["document"]
            summary = entry["summary"]

        if self.contains_entity:
            summ_entities = entry["summ_entities"]
            summ_ent_pos = entry["summ_ent_pos"]
            doc_entities = entry["doc_entities"]
            doc_ent_pos = entry["doc_ent_pos"]
        else:
            summ_entities, summ_ent_pos = get_entities(self.spacy_model, summary)
            doc_entities, doc_ent_pos = get_entities(self.spacy_model, document)

        # get input ids and offset mapping for document
        out = self.tokenizer(
            document,
            truncation=True,
            max_length=self.max_length_input,
            return_offsets_mapping=True,
        )
        input_ids = out["input_ids"]
        char_token_mapping = out["offset_mapping"]
        # input_ent_mapping: (len_input * num_ent)
        doc_entities, doc_ent_pos = self.cut_entity_list(
            doc_entities, doc_ent_pos, char_token_mapping[-2][1]
        )
        char_token_mapping_new = [
            (m[0] + 1, m[1]) if document[m[0]] == " " else m
            for m in char_token_mapping[:-1]
        ]
        char_token_mapping_new.append(char_token_mapping[-1])
        char_token_mapping = char_token_mapping_new
        input_ent_mapping = self.map_entities_to_tokens(
            input_ids, char_token_mapping, doc_ent_pos
        )
        # pdb.set_trace()
        # get input ids and offset mapping for summaries
        out = self.tokenizer(
            summary,
            truncation=True,
            max_length=self.max_length_output,
            return_offsets_mapping=True,
        )
        output_ids = out["input_ids"]
        char_token_mapping = out["offset_mapping"]
        char_token_mapping_new = [
            (m[0] + 1, m[1]) if summary[m[0]] == " " else m
            for m in char_token_mapping[:-1]
        ]
        char_token_mapping_new.append(char_token_mapping[-1])
        char_token_mapping = char_token_mapping_new
        summ_entities, summ_ent_pos = self.cut_entity_list(
            summ_entities, summ_ent_pos, char_token_mapping[-2][1]
        )
        input_output_mapping = self.map_entity_input_output(summ_entities, doc_entities)
        # pdb.set_trace()
        # pdb.set_trace()
        # output_ent_mapping: (len_input * len_input_with_ent)
        # if summary[char_token_mapping[-2][1]] in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        # char_token_mapping[-2][1] -= 1  # ignore the terminal symbol.
        (
            output_ent_mapping,
            labels,
        ) = self.map_orig_output_to_entity_output_and_build_labels(
            output_ids, char_token_mapping, summ_ent_pos, input_output_mapping
        )

        # for pegasus
        output_ids = [self.tokenizer.pad_token_id] + output_ids[
            :-1
        ]  # remove the last token, i.e. </s> token
        new_output_ent_mapping = torch.zeros(output_ent_mapping.shape)
        new_output_ent_mapping[0, 0] = 1
        new_output_ent_mapping[1:, 1:] = output_ent_mapping[:-1, :-1]
        output_ent_mapping = new_output_ent_mapping

        return (
            torch.tensor(input_ids),
            torch.tensor(output_ids),
            torch.tensor(labels),
            input_ent_mapping,
            output_ent_mapping,
        )

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        # data = self.process_single_data(entry)
        try:
            data = self.process_single_data(entry)
            data += (str(idx),)
        except:
            data = None
            print("%d is wrong" % (idx))
        return data


class TopicModelDataset(Dataset):
    def __init__(self, dataset, vectorizer):
        self.dataset = dataset
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        document = self.dataset[idx]
        d_bow = self.vectorizer.transform([document]).toarray()
        d_bow = d_bow.astype(np.float32)
        return d_bow[0]


class TopicModelDatasetMXM(Dataset):
    def __init__(self, hf_dataset, vocabulary):
        self.hf_dataset = hf_dataset
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        document = self.hf_dataset[idx]
        return document


class SummGuidedTopicModelDataset(Dataset):
    def __init__(self, hf_dataset, vectorizer):
        self.hf_dataset = hf_dataset
        self.vectorizer = vectorizer

    # def fit(self)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        document = entry["article"]
        summary = entry["highlights"]
        out = self.vectorizer.transform([document, summary])
        d_bow = out[0].toarray()
        d_bow = d_bow.astype(np.float32)
        s_bow = out[1].toarray()
        s_bow = s_bow.astype(np.float32)
        # input_ids = self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.tokenize(document)
        # )
        # output_ids = self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.tokenize(summary)
        # )
        # d_bow = torch.zeros(self.tokenizer.vocab_size)
        # for wid in input_ids:
        #     d_bow[wid] += 1
        # s_bow = torch.zeros(self.tokenizer.vocab_size)
        # for wid in output_ids:
        #     s_bow[wid] += 1
        return d_bow[0], s_bow[0]


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    # pad_token_id = 0  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
    batch = list(filter(lambda x: x is not None, batch))
    input_ids, output_ids = list(zip(*batch))
    if input_ids[0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif input_ids[0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )

    return input_ids, output_ids


def collate_fn_with_ent(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    pad_token_id = 0  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id

    input_ids, output_ids, summ_ent = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    summ_ent = torch.stack(summ_ent, 0)

    return input_ids, output_ids, summ_ent


def collate_fn_spanCopy(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    # pad_token_id = 0  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
    batch = list(filter(lambda x: x is not None, batch))
    input_ids, output_ids, labels, input_ent_mapping, output_ent_mapping, fid = list(
        zip(*batch)
    )
    if input_ids[0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif input_ids[0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=pad_token_id
    )
    input_ent_mapping_batch = torch.zeros(
        len(input_ent_mapping),
        max([i.shape[0] for i in input_ent_mapping]),
        max([i.shape[1] for i in input_ent_mapping]),
    )
    output_ent_mapping_batch = torch.zeros(
        len(output_ent_mapping),
        max([i.shape[0] for i in output_ent_mapping]),
        max([i.shape[1] for i in output_ent_mapping]),
    )
    for i in range(len(output_ent_mapping)):
        input_ent_mapping_batch[
            i, : input_ent_mapping[i].shape[0], : input_ent_mapping[i].shape[1]
        ] = input_ent_mapping[i]
        output_ent_mapping_batch[
            i, : output_ent_mapping[i].shape[0], : output_ent_mapping[i].shape[1]
        ] = output_ent_mapping[i]
    return (
        input_ids,
        input_ent_mapping_batch,
        output_ids,
        labels,
        output_ent_mapping_batch,
        fid,
    )


# def build_vocabulary(all_docs,vocabulary_size,saveas=None):
#     vectorizer = CountVectorizer(max_features=vocabulary_size)
#     vectorizer.fit_transform(all_docs)
#     word= list(vectorizer.get_feature_names_out())

#     if saveas!=None:

#     return vocab

if __name__ == "__main__":
    # data_path = "/scratch/xiaowen3/dataset/cnndm_with_ent.pt"
    # print("load tokenizer")
    # tokenizer = PegasusTokenizerFast.from_pretrained(
    #     "/project/def-carenini/xiaowen3/pretrained_models/pegasus-cnndm/"
    # )
    # print("load dataset")
    from tqdm import tqdm

    data_path = "/scratch/xiaowen3/dataset/cnndm_with_ent/train.pt"
    # data_path = "/scratch/xiaowen3/dataset/xsum_with_ent_filtered_noempty/test.pt"
    print("load tokenizer")
    # tokenizer = PegasusTokenizerFast.from_pretrained(
    #     "google/pegasus-cnn_dailymail",
    #     cache_dir="/scratch/wenxiao/pretrained_models/pegasus",
    # )
    tokenizer = PegasusTokenizerFast.from_pretrained(
        "/project/def-carenini/xiaowen3/pretrained_models/pegasus-xsum-pretrained"
    )
    print("load dataset")
    dataset = torch.load(data_path)
    d = spanCopySummDataset(
        dataset, "cnndm", tokenizer, max_length_input=512, max_length_output=64,
    )
    all_ent_num = []
    wrong = 0
    # print(d[39])
    for b in tqdm(range(len(d))):
        # a = d[b]
        # labels = a[2]
        # all_ent_num.append((labels >= tokenizer.vocab_size).sum())
        a = d[b]
    # continue
    avg = sum(all_ent_num) / len(all_ent_num)
    print("avg is %f" % (avg))
    print("wrong data number is %d" % (wrong))
    # a = d[2891]
    # if a is None:
    #     break
    # train_dataloader = DataLoader(
    #     d, batch_size=1, shuffle=True, collate_fn=collate_fn_spanCopy, num_workers=2,
    # )
    # print("start iteration")
    # for b in train_dataloader:
    #     a = b

