import pandas as pd
import numpy as np
# from transformers import BertModel, BertConfig, AutoTokenizer
from transformers import DistilBertModel, ElectraModel, ElectraForTokenClassification, AutoTokenizer
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


MAX_SEQ_LEN = 512


class BertClassifierBinary(nn.Module):
    def __init__(self):
        super().__init__()
        # we use autoTokenizer here as BertTokenizer will call the tokenizer of pytorch,
        # where encode_plus() is not available
        self.bertTokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', normalization=True)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 2)  # 2 classes

    def tokenizer(self):
        return self.bertTokenizer

    def forward(self, input_seq, mask):
        last_hidden_state = self.bert(input_ids=input_seq, attention_mask=mask)[0]
        probabilities = self.classifier(last_hidden_state)
        # probabilities = self.classifier(self.dropout(last_hidden_state))
        return probabilities


class BertClassifierMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.bertTokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', normalization=True)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 144)  # 144 classes

    def tokenizer(self):
        return self.bertTokenizer

    def forward(self, input_seq, mask):
        last_hidden_state = self.bert(input_ids=input_seq, attention_mask=mask)[0]
        probabilities = self.classifier(last_hidden_state)
        # probabilities = self.classifier(self.dropout(last_hidden_state))
        return probabilities


class ElectraMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.electraTokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', normalization=True)
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        # self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.electra.config.hidden_size, 144)  # 144 classes

    def tokenizer(self):
        return self.electraTokenizer

    def forward(self, input_seq, mask):
        output = self.electra(input_ids=input_seq, attention_mask=mask)[0]
        probabilities = self.classifier(output)
        # probabilities = self.classifier(self.dropout(output))
        return probabilities


# class ElectraBinary(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.electraTokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', normalization=True)
#         self.electra = ElectraForTokenClassification.from_pretrained('google/electra-small-discriminator')
#         # self.dropout = nn.Dropout(p=0.1)
#
#     def tokenizer(self):
#         return self.electraTokenizer
#
#     def forward(self, input_seq, mask):
#         output = self.electra(input_ids=input_seq, attention_mask=mask)
#         return output.logits


class ElectraBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.electraTokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', normalization=True)
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        # self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.electra.config.hidden_size, 2)  # 2 classes

    def tokenizer(self):
        return self.electraTokenizer

    def forward(self, input_seq, mask):
        output = self.electra(input_ids=input_seq, attention_mask=mask)[0]
        probabilities = self.classifier(output)
        # probabilities = self.classifier(self.dropout(output))
        return probabilities


# data: a list of dictionaries, where the dictionaries include pn_num, input_seq, mask, labels
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


def prepare_f1(predictions, labels):
    tp = np.sum((predictions == labels) & (predictions != 0))
    fp = np.sum((predictions != 0) & (labels == 0))
    tn = np.sum((predictions == labels) & (predictions == 0))
    fn = np.sum((predictions == 0) & (labels != 0))
    return tp, fp, tn, fn


def prepare_f1_keyinfo(predictions, labels, keyinfo_idx):
    tp = np.sum((predictions == labels) & (predictions == keyinfo_idx))
    fp = np.sum((labels != keyinfo_idx) & (predictions == keyinfo_idx))
    tn = np.sum((labels != keyinfo_idx) & (predictions != keyinfo_idx))
    fn = np.sum((labels == keyinfo_idx) & (predictions != keyinfo_idx))
    return tp, fp, tn, fn


def calculate_micro_f1(tp, fp, tn, fn):
    f1 = tp/(tp+0.5*(fp+fn))
    return f1


def customize_train_test_split(preprocessed_data, method='Multi'):
    train_val_set = []
    test_set = []
    if method == 'Multi':  # group by case
        case_sample_set = [[] for _ in range(10)]
        for data_sample in preprocessed_data:
            case_sample_set[data_sample['case_num']].append(data_sample)
        for case_set in case_sample_set:
            train_samples, test_samples = train_test_split(case_set, test_size=0.2, random_state=42, shuffle=True)
            train_val_set += train_samples
            test_set += test_samples
    else:  # group by keyinfo
        key_sample_set = [[] for _ in range(143)]
        key_num_idx_map, _ = key_idx_num_map()
        for data_sample in preprocessed_data:
            key_sample_set[key_num_idx_map[str(data_sample['keyinfo_num'])]-1].append(data_sample)
        for key_set in key_sample_set:
            train_samples, test_samples = train_test_split(key_set, test_size=0.2, random_state=42, shuffle=True)
            train_val_set += train_samples
            test_set += test_samples
    return train_val_set, test_set


def dataloaders(preprocessed_data, batch_size, method='Multi'):
    train_val_set, test_set = customize_train_test_split(preprocessed_data, method=method)
    train_set, val_set = train_test_split(train_val_set, test_size=0.1, random_state=42, shuffle=True)

    train_dataset = OurDataset(train_set)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = OurDataset(val_set)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_dataset = OurDataset(test_set)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def preprocess_multi(tokenizer, batch_size):
    origin_data = pd.read_csv("train.csv", header=0)
    origin_pn_notes = pd.read_csv("patient_notes.csv", header=0)

    classes = origin_data['feature_num'].unique().tolist()
    classes.insert(0, -1)  # where (-1) represent no annotation detected

    # origin_pn_notes : pn_history
    annoted_pn_nums = origin_data['pn_num'].unique()
    # pn_num - history dictionary
    pn_num_his_dic = dict()
    for pn_num in annoted_pn_nums:
        pn_num_his_dic[pn_num] = origin_pn_notes.loc[origin_pn_notes['pn_num'] == pn_num].pn_history.item()

    labelEncoder = LabelEncoder()
    labelEncoder.fit(classes)

    origin_data['encoded_feature_num'] = labelEncoder.transform(origin_data['feature_num'])
    # print(origin_data['encoded_feature_num'].unique())

    # with use 0 for no annotation in the following code block
    preprocessed_data = []
    for pn_anno_group in origin_data.groupby(['pn_num']):
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

    # we will use 200 pn_note for test and 800 for train_val
    # in which 720 for training, 80 for validation (in case overfitting)

    return dataloaders(preprocessed_data, batch_size, method='Multi')


def preprocess_binary(tokenizer, batch_size):
    preprocessed_data = []

    origin_data = pd.read_csv("train.csv", header=0)
    origin_pn_notes = pd.read_csv("patient_notes.csv", header=0)
    feature_data = pd.read_csv("features.csv", header=0)

    # origin_pn_notes : pn_history
    annoted_pn_nums = origin_data['pn_num'].unique()
    # pn_num - pn_history dictionary
    pn_num_his_dic = dict()
    for pn_num in annoted_pn_nums:
        pn_num_his_dic[pn_num] = origin_pn_notes.loc[origin_pn_notes['pn_num'] == pn_num].pn_history.item()

    for _, row in origin_data.iterrows():
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

    # we will use 2860 annotation records for test, and 11440 annotation records for train_val,
    # in which 10296 for training, 1144 for validation (in case overfitting)

    return dataloaders(preprocessed_data, batch_size, method='Binary')


def train_pipeline(model, optimizer, criterion, device, train_loader, num_classes):
    model.train()
    train_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for idx, data in enumerate(train_loader):
        seq, mask, label, _ = data
        # transit to one-hot
        label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)
        seq = torch.Tensor(seq).to(torch.int32).to(device)
        mask = torch.Tensor(mask).to(torch.uint8).to(device)
        # label = one_hot_label_smoothing(label)
        label = label.to(device)
        optimizer.zero_grad()

        # with autocast():
        #     prediction = model(seq, mask)
        #     loss = criterion(torch.permute(prediction, (0, 2, 1)), torch.argmax(label, dim=2))

        prediction = model(seq, mask)
        # prediction = F.softmax(prediction, dim=2)
        loss = criterion(torch.permute(prediction, (0, 2, 1)), torch.argmax(label, dim=2))

        # loss = criterion(prediction, label.to(torch.float32))

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        train_loss += loss.item()
        tp_add, fp_add, tn_add, fn_add = prepare_f1(torch.argmax(prediction, dim=2).detach().cpu().numpy(),
                                                    torch.argmax(label, dim=2).detach().cpu().numpy())
        tp += tp_add
        fp += fp_add
        tn += tn_add
        fn += fn_add
    f1_score = calculate_micro_f1(tp, fp, tn, fn)
    train_loss = train_loss / len(train_loader)
    return train_loss, f1_score


def val_test_pipeline(model, criterion, device, val_test_loader, num_classes):
    model.eval()
    val_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for idx, data in enumerate(val_test_loader):
            seq, mask, label, record_info = data
            # transit to one-hot
            label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)
            seq = torch.Tensor(seq).to(torch.int32).to(device)
            mask = torch.Tensor(mask).to(torch.uint8).to(device)
            label = label.to(device)

            prediction = model(seq, mask)
            # prediction = F.softmax(prediction, dim=2)
            loss = criterion(torch.permute(prediction, (0, 2, 1)), torch.argmax(label, dim=2))

            val_loss += loss.item()
            tp_add, fp_add, tn_add, fn_add = prepare_f1(torch.argmax(prediction, dim=2).detach().cpu().numpy(),
                                                        torch.argmax(label, dim=2).detach().cpu().numpy())
            tp += tp_add
            fp += fp_add
            tn += tn_add
            fn += fn_add
    f1_score = calculate_micro_f1(tp, fp, tn, fn)
    val_loss = val_loss / len(val_test_loader)
    return val_loss, f1_score

def eval_pipeline(model, device, test_loader, method):
    model.eval()
    predictions = []
    num_classes = 144 if method == 'Multi' else 2
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            seq, mask, label, record_info = data
            # transit to one-hot
            label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)
            seq = torch.Tensor(seq).to(torch.int32).to(device)
            mask = torch.Tensor(mask).to(torch.uint8).to(device)
            label = label.to(device)
            prediction = model(seq, mask)
            predictions.append(torch.argmax(prediction, dim=2).detach().cpu().numpy())
    
    return post_processing(predictions, method, test_loader)


def post_processing(predictions, method, test_loader):
    key_info_nums = []
    pn_nums = []
    detected_character_span = []
    if method == "Multi":
        _, keyinfo_idx_num_map = key_idx_num_map()
        
        for idx, data in enumerate(test_loader):
            seq, mask, label, record_info = data
            character_spans = record_info['character_span']
            prediction = predictions[idx].tolist()[0]
            for key_idx in range(1, 144):
                detect_span = []
                key_num = keyinfo_idx_num_map[key_idx]
                indexes = [i for i in range(record_info['origin_token_length']) if prediction[i] == key_idx]
                for index in indexes:
                    detached_character_span = [character_idx.detach().cpu().numpy()[0] for character_idx in character_spans[index]]
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
                detached_character_span = [character_idx.detach().cpu().numpy()[0] for character_idx in character_spans[index]]
                detect_span.append(detached_character_span)
            detected_character_span.append(detect_span)
            key_info_nums.append(record_info['keyinfo_num'].detach().cpu().numpy()[0])
            pn_nums.append(record_info['pn_num'].detach().cpu().numpy()[0])
    return detected_character_span, key_info_nums, pn_nums


def integrate_character_span(char_span):
    if not char_span:
        return []
    merged_tuples = []
    current_tuple = char_span[0]
    for i in range(1, len(char_span)):
        if current_tuple[1] + 1 == char_span[i][0] or current_tuple[1] == char_span[i][0]:
            current_tuple = [current_tuple[0], char_span[i][1]]
        else:
            merged_tuples.append(current_tuple)
            current_tuple = char_span[i]
    merged_tuples.append(current_tuple)
    return merged_tuples


def form_out_character_span(char_span):
    new_out = []
    for span in char_span:
        new_out.append(integrate_character_span(span))
    return new_out


def key_idx_num_map():
    keyinfo_num_lst = pd.read_csv("features.csv", header=0)['feature_num'].unique()
    keyinfo_num_idx_map = {}
    keyinfo_idx_num_map = {}
    for i in range(len(keyinfo_num_lst)):
        keyinfo_num_idx_map[str(keyinfo_num_lst[i])] = i+1
        keyinfo_idx_num_map[str(i+1)] = keyinfo_num_lst[i]
    return keyinfo_num_idx_map, keyinfo_idx_num_map


def case_key_map():
    case_key_info_map = {}
    info_df = pd.read_csv("features.csv", header=0)
    for case, keyinfo_list in info_df.groupby(['case_num']):
        case_key_info_map[str(case)] = keyinfo_list['feature_num'].unique()
    return case_key_info_map


def test_macros(model, device, test_loader, num_classes, type='Multi'):
    model.eval()
    case_lst = {str(key): [] for key in range(10)}
    keyinfo_num_lst = pd.read_csv("features.csv", header=0)['feature_num'].unique()
    keyinfo_lst = {str(key): [] for key in keyinfo_num_lst}
    keyinfo_num_idx_map, _ = key_idx_num_map()
    case_key_info_map = case_key_map()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            seq, mask, label, record_info = data
            # transit to one-hot
            label = F.one_hot(torch.Tensor(label).to(torch.int64), num_classes=num_classes)
            seq = torch.Tensor(seq).to(torch.int32).to(device)
            mask = torch.Tensor(mask).to(torch.uint8).to(device)
            label = label.to(device)

            prediction = model(seq, mask)
            # prediction = F.softmax(prediction, dim=2)

            prediction_cpu = torch.argmax(prediction, dim=2).detach().cpu().numpy()
            labels_cpu = torch.argmax(label, dim=2).detach().cpu().numpy()

            tp_add, fp_add, tn_add, fn_add = prepare_f1(prediction_cpu, labels_cpu)

            if type == 'Multi':
                record_f1_score = calculate_micro_f1(tp_add, fp_add, tn_add, fn_add)
                case_lst[str(record_info['case_num'].item())].append(record_f1_score)
                keyinfo_to_check = case_key_info_map[str(record_info['case_num'].item())]
                for keyinfo_num in keyinfo_to_check:
                    keyinfo_idx = keyinfo_num_idx_map[str(keyinfo_num)]
                    # calculate f1 score for current key info
                    if np.sum((labels_cpu == keyinfo_idx)) > 0:
                        tp_key, fp_key, tn_key, fn_key = prepare_f1_keyinfo(prediction_cpu, labels_cpu, keyinfo_idx)
                        current_keyinfo_f1 = calculate_micro_f1(tp_key, fp_key, tn_key, fn_key)
                        keyinfo_lst[str(keyinfo_num)].append(current_keyinfo_f1)
            else:
                if np.sum((labels_cpu == 1)) > 0:
                    record_f1_score = calculate_micro_f1(tp_add, fp_add, tn_add, fn_add)
                    keyinfo_lst[str(record_info['keyinfo_num'].item())].append(record_f1_score)
                    case_lst[str(record_info['case_num'].item())].append(record_f1_score)
    case_f1_macro = {}
    key_f1_macro = {}
    for key in range(10):
        if len(case_lst[str(key)]) > 0:
            case_f1_macro[str(key)] = sum(case_lst[str(key)]) / len(case_lst[str(key)])
        else:
            case_f1_macro[str(key)] = 0  # "Not_detect"
    for key in keyinfo_lst.keys():
        if len(keyinfo_lst[key]) > 0:
            key_f1_macro[key] = sum(keyinfo_lst[key]) / len(keyinfo_lst[key])
        else:
            key_f1_macro[key] = 0  # "Not_detect"

    return case_f1_macro, key_f1_macro


def one_hot_label_smoothing(labels, scaler=0.2):
    num_class = labels.shape[2]
    smoothed = (1 - scaler) * labels + torch.tensor(scaler/num_class)
    return smoothed
