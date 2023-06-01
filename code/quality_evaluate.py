import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from ModelsFuncs import BertClassifierMulti, ElectraBinary, BertClassifierBinary, ElectraMulti, key_idx_num_map
from ModelsFuncs import form_out_character_span
import numpy as np


BATCH_SIZE = 16
DEVICE = "cuda"
MAX_SEQ_LEN = 512


class OurDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record_info = {
            'pn_num': self.data[index]['pn_num'],
            'keyinfo_num': self.data[index]['keyinfo_num'],
            'case_num': self.data[index]['case_num'],
            'origin_token_length': self.data[index]['origin_token_length'],
            'character_span': self.data[index]['character_span']
        }
        seq = np.array(self.data[index]['input_seq'])
        masks = np.array(self.data[index]['mask'])
        labels = np.array(self.data[index]['labels'])
        return seq, masks, labels, record_info


def preprocess_multi_dataloader(train_df, pn_df, eval_df, tokenizer):
    classes = train_df['feature_num'].unique().tolist()
    classes.insert(0, -1)  # where (-1) represent no annotation detected

    # origin_pn_notes : pn_history
    annoted_pn_nums = eval_df['pn_num'].unique()
    # pn_num - history dictionary
    pn_num_his_dic = dict()
    for pn_num in annoted_pn_nums:
        pn_num_his_dic[pn_num] = pn_df.loc[pn_df['pn_num'] == pn_num].pn_history.item()

    labelEncoder = LabelEncoder()
    labelEncoder.fit(classes)

    eval_df['encoded_feature_num'] = labelEncoder.transform(eval_df['feature_num'])
    # print(origin_data['encoded_feature_num'].unique())

    # with use 0 for no annotation in the following code block
    preprocessed_data = []
    for pn_anno_group in eval_df.groupby(['pn_num']):
        pn_num_data_dic = {}
        pn_note = pn_num_his_dic[pn_anno_group[0]]
        tokens = tokenizer.encode_plus(pn_note, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True,
                                       return_offsets_mapping=True)
        origin_token_length = len(tokenizer.encode_plus(pn_note)['input_ids'])
        case_num = -1
        seq = tokens['input_ids']
        mask = tokens['attention_mask']
        character_spans = tokens['offset_mapping']
        labels = [0] * 512  # init as empty annotation
        # assign label based on annotation

        df = pn_anno_group[1]
        cls_loc_dic = dict()
        for _, row in df.iterrows():
            # update case number
            if case_num == -1:
                case_num = row['case_num']

            f_num = row['encoded_feature_num']
            location = row['location'].replace("[", "").replace("]", "").replace("'", "").replace(";", ", ")
            location = location.split(', ')
            # remove no-annotation cases
            if location[0] != '':
                spans = [group.split(' ') for group in location]
                cls_loc_dic[f_num] = spans

        for idx, span in enumerate(character_spans):
            # if span is [cls], [seq], paddings, then jump
            if span == (0, 0):
                continue
            find = False
            for key in cls_loc_dic.keys():
                if find:
                    break
                annos = cls_loc_dic[key]
                for anno in annos:
                    if find:
                        break
                    else:
                        if span[0] >= int(anno[0]) and span[1] <= int(anno[1]):
                            find = True
                            labels[idx] = key

        pn_num_data_dic['pn_num'] = pn_anno_group[0]
        pn_num_data_dic['keyinfo_num'] = -1
        pn_num_data_dic['case_num'] = case_num
        pn_num_data_dic['input_seq'] = seq
        pn_num_data_dic['mask'] = mask
        pn_num_data_dic['labels'] = labels
        pn_num_data_dic['character_span'] = character_spans
        pn_num_data_dic['origin_token_length'] = origin_token_length
        preprocessed_data.append(pn_num_data_dic)

    test_set = OurDataset(preprocessed_data)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    return test_dataloader


def preprocess_binary_dataloader(pn_df, eval_df, tokenizer):
    preprocessed_data = []

    feature_data = pd.read_csv("features.csv", header=0)

    # origin_pn_notes : pn_history
    annoted_pn_nums = eval_df['pn_num'].unique()
    # pn_num - pn_history dictionary
    pn_num_his_dic = dict()
    for pn_num in annoted_pn_nums:
        pn_num_his_dic[pn_num] = pn_df.loc[pn_df['pn_num'] == pn_num].pn_history.item()

    for _, row in eval_df.iterrows():
        pn_num_data_dic = {}
        pn_note = pn_num_his_dic[row['pn_num']]  # patient note
        info_str = ". @ " + feature_data.loc[feature_data['feature_num'] == row['feature_num']][
            'feature_text'].item().replace('-', ' ')  # the key information text with @
        # Since the longest token list of patient notes is 282 << 512,
        # and the length of token list for key info is less than 100,
        # so we can concatenate the string first then tokenize
        pn_info = pn_note + info_str
        pn_info_token = tokenizer.encode_plus(pn_info, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True,
                                              return_offsets_mapping=True)
        origin_token_length = len(tokenizer.encode_plus(pn_note)['input_ids'])
        seq = pn_info_token['input_ids']
        mask = pn_info_token['attention_mask']
        character_spans = pn_info_token['offset_mapping']

        labels = [0] * 512
        # location of the answer in the patient note
        location = row['location'].replace("[", "").replace("]", "").replace("'", "").replace(";", ", ").split(', ')
        token_spans = [group.split(' ') for group in location] if location[0] != '' else []  # label span

        for idx, span in enumerate(character_spans):
            # if span is [cls], [seq], paddings, then jump
            if span != (0, 0):
                for anno_span in token_spans:
                    if anno_span and span[0] >= int(anno_span[0]) and span[1] <= int(anno_span[1]):
                        labels[idx] = 1
                        break

        pn_num_data_dic['input_seq'] = seq
        pn_num_data_dic['pn_num'] = row['pn_num']
        pn_num_data_dic['keyinfo_num'] = row['feature_num']
        pn_num_data_dic['case_num'] = row['case_num']
        pn_num_data_dic['mask'] = mask
        pn_num_data_dic['labels'] = labels
        pn_num_data_dic['character_span'] = character_spans
        pn_num_data_dic['origin_token_length'] = origin_token_length
        preprocessed_data.append(pn_num_data_dic)

    test_set = OurDataset(preprocessed_data)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    return test_dataloader


def post_processing(predictions, method, test_loader, key_info_to_detect=[]):
    key_info_nums = []
    pn_nums = []
    detected_character_span = []
    if method == "Multi":
        keyinfo_num_idx_map, _ = key_idx_num_map()
        for idx, data in enumerate(test_loader):
            seq, mask, label, record_info = data
            character_spans = record_info['character_span']
            prediction = predictions[idx].tolist()[0]
            for key_num in key_info_to_detect:
                detect_span = []
                key_idx = keyinfo_num_idx_map[str(key_num)]
                indexes = [i for i in range(record_info['origin_token_length']) if prediction[i] == key_idx]
                for index in indexes:
                    detached_character_span = [character_idx.detach().cpu().numpy()[0] for character_idx in
                                               character_spans[index]]
                    detect_span.append(detached_character_span)
                detected_character_span.append(detect_span)
                key_info_nums.append(key_num)
                pn_nums.append(record_info['pn_num'].detach().cpu().numpy()[0])
    else:
        for idx, data in enumerate(test_loader):
            detect_span = []
            seq, mask, label, record_info = data
            character_spans = record_info['character_span']
            prediction = predictions[idx].tolist()[0]
            indexes = [i for i in range(record_info['origin_token_length']) if prediction[i] == 1]
            for index in indexes:
                detached_character_span = [character_idx.detach().cpu().numpy()[0] for character_idx in
                                           character_spans[index]]
                detect_span.append(detached_character_span)
            detected_character_span.append(detect_span)
            key_info_nums.append(record_info['keyinfo_num'].detach().cpu().numpy()[0])
            pn_nums.append(record_info['pn_num'].detach().cpu().numpy()[0])
    return detected_character_span, key_info_nums, pn_nums



def eval_pipeline(model, device, method):
    if method == "Multi":
        test_loader = preprocess_multi_dataloader(origin_data, origin_pn_notes, df_multi, model.tokenizer())
    else:
        test_loader = preprocess_binary_dataloader(origin_pn_notes, df_binary, model.tokenizer())

    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            seq, mask, label, record_info = data
            # transit to one-hot
            seq = torch.Tensor(seq).to(torch.int32).to(device)
            mask = torch.Tensor(mask).to(torch.uint8).to(device)

            prediction = model(seq, mask)
            prediction = torch.argmax(prediction, dim=2).detach().cpu().numpy()
            predictions.append(prediction)
    return predictions, test_loader


if __name__ == '__main__':
    device = "cuda"
    origin_data = pd.read_csv("train.csv", header=0)
    origin_pn_notes = pd.read_csv("patient_notes.csv", header=0)

    df_binary = origin_data[(origin_data["feature_num"] == 403) | (origin_data["feature_num"] == 704) |
                            (origin_data["feature_num"] == 705)]
    pn_note_nums = df_binary["pn_num"].unique()
    df_multi = origin_data[origin_data["pn_num"].isin(pn_note_nums)]

    models = []
    model_configs = []
    TRAIN = False

    model_1 = BertClassifierMulti()
    model_config_1 = {
        "used_model": "DistilBert",
        "method": "Multi",
        "num_classes": 144
    }
    models.append(model_1)
    model_configs.append(model_config_1)

    model_2 = ElectraMulti()
    model_config_2 = {
        "used_model": "ELECTRA",
        "method": "Multi",
        "num_classes": 144
    }
    models.append(model_2)
    model_configs.append(model_config_2)

    model_3 = BertClassifierBinary()
    model_config_3 = {
        "used_model": "DistilBert",
        "method": "Binary",
        "num_classes": 2
    }
    models.append(model_3)
    model_configs.append(model_config_3)

    model_4 = ElectraBinary()
    model_config_4 = {
        "used_model": "ELECTRA",
        "method": "Binary",
        "num_classes": 2
    }
    models.append(model_4)
    model_configs.append(model_config_4)

    case_f1_macro_lst = []
    key_f1_macro_lst = []
    predictions_lst = []
    test_loader_lst = []

    for i in range(4):
        model_name_final = model_configs[i]["used_model"] + "_" + model_configs[i]["method"] + "_nbme_bert_v2.pth"
        weights = torch.load(model_name_final)
        models[i].load_state_dict(weights)

        predictions, test_loader = eval_pipeline(models[i], device, model_configs[i]["method"])

        detected_character_spans, key_info_nums, pn_nums = post_processing(predictions, model_configs[i]["method"],
                                                                           test_loader, [403, 704, 705])

        # df_dic = {
        #     "pn_num": pn_nums,
        #     "key_info_num": key_info_nums,
        #     "character_span": detected_character_spans
        # }
        #
        # df_new = pd.DataFrame(df_dic)
        # file_name = model_configs[i]["used_model"] + "_" + model_configs[i]["method"]
        # df_new.to_csv("{}.csv".format(file_name))

        df_dic = {
            "pn_num": pn_nums,
            "key_info_num": key_info_nums,
            "character_span": form_out_character_span(detected_character_spans)
        }

        df_postprocessed = pd.DataFrame(df_dic)
        file_name = model_configs[i]["used_model"] + "_" + model_configs[i]["method"]
        df_postprocessed.to_csv("{}.csv".format(file_name))
