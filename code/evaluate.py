from ModelsFuncs import BertClassifierBinary, BertClassifierMulti, ElectraMulti, ElectraBinary
from ModelsFuncs import preprocess_multi, preprocess_binary, train_pipeline, val_test_pipeline, test_macros
from ModelsFuncs import key_idx_num_map
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
import matplotlib.pyplot as plt


BATCH_SIZE = 16
DEVICE = "cuda"


def preprocess(method, tokenizer, batch_size):
    if method == "Multi":
        return preprocess_multi(tokenizer=tokenizer, batch_size=batch_size)
    else:
        return preprocess_binary(tokenizer=tokenizer, batch_size=batch_size)


if __name__ == '__main__':
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

    micro_f1_list = []
    case_f1_list = []
    key_f1_list = []
    case_f1_detail_list = []
    key_f1_detail_list = []
    time_list = []
    time_per_sample_lst = []

    criterion = nn.CrossEntropyLoss()

    for i in range(4):
        model = models[i]
        tokenizer = model.tokenizer()
        model.to(DEVICE)
        method = model_configs[i]["method"]

        train_loader, val_loader, test_loader = preprocess(method=method, tokenizer=tokenizer, batch_size=BATCH_SIZE)
        model_name_final = model_configs[i]["used_model"] + "_" + method + "_nbme_bert_v2.pth"
        weights = torch.load(model_name_final)
        model.load_state_dict(weights)

        print(model_configs[i]["used_model"], "_", method, " evaluation start:")
        total_samples = len(test_loader) * BATCH_SIZE
        print("total test cases: ", total_samples)
        test_start = time.time()
        _, test_f1_score = val_test_pipeline(model, criterion, DEVICE, test_loader, model_configs[i]["num_classes"])
        test_end = time.time()
        total_time = test_end - test_start
        print("Test: f1 ", test_f1_score, " total time:", total_time, " per sample time:", total_time/total_samples)
        micro_f1_list.append(test_f1_score)
        time_list.append(total_time)
        time_per_sample_lst.append(total_time/total_samples)

        case_f1_macro, keyinfo_lst = test_macros(model, DEVICE, test_loader, model_configs[i]["num_classes"], type=method)
        print(case_f1_macro)
        print("case_f1: ", sum(case_f1_macro.values()) / 10)
        print(keyinfo_lst)
        print("key_f1: ", sum(keyinfo_lst.values()) / 143)
        case_f1_list.append(sum(case_f1_macro.values()) / 10)
        case_f1_detail_list.append(case_f1_macro.values())
        key_f1_list.append(sum(keyinfo_lst.values()) / 143)
        key_f1_detail_list.append(keyinfo_lst.values())

    # plot fig for case f1 score
    # overview
    x_case = [case for case in range(10)]
    fig_1, ax_1 = plt.subplots()
    for i in range(4):
        label = model_configs[i]["used_model"] + "_" + model_configs[i]["method"]
        ax_1.scatter(case_f1_detail_list[i], x_case, label=label)
        ax_1.set_ylabel('Medical Case')
        ax_1.set_xlabel('Case F1 Score')
        ax_1.set_title('Case F1 Score Overview')
    ax_1.legend(loc='lower left')

    fig_1.savefig('case f1 score_overview.png')

    # separate
    for i in range(4):
        fig, ax = plt.subplots()
        fig_name = model_configs[i]["used_model"] + "_" + model_configs[i]["method"]
        ax.bar(x_case, case_f1_detail_list[i])
        ax.set_xlabel('Medical Case')
        ax.set_ylabel('Case F1 Score')
        ax.set_title(fig_name)
        fig.savefig('{} case F1.png'.format(fig_name))

    # plot fig for key f1 score
    # overview
    x_key = key_idx_num_map()[0].keys()
    fig_2, ax_2 = plt.subplots()
    for i in range(4):
        label = model_configs[i]["used_model"] + "_" + model_configs[i]["method"]
        ax_2.scatter(key_f1_detail_list[i], x_key, label=label)
        ax_2.set_ylabel('Key Info')
        ax_2.set_xlabel('Key F1 Score')
        ax_2.set_title('Key F1 Score Overview')
    ax_2.legend()

    fig_2.savefig('Key f1 score_overview.png')
