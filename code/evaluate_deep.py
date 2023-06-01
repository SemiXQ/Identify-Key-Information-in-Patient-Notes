from ModelsFuncs import BertClassifierBinary, BertClassifierMulti, ElectraMulti, ElectraBinary
from ModelsFuncs import preprocess_multi, preprocess_binary, train_pipeline, val_test_pipeline, test_macros
from ModelsFuncs import key_idx_num_map, case_key_map
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
        key_f1_detail_list.append(keyinfo_lst)

    # plot key f1 score fig per case per model
    keyinfo_num_idx_map, _ = key_idx_num_map()
    for m in range(4):
        scores = key_f1_detail_list[m]
        for case in range(10):
            key_info_to_plot = case_key_map()[str(case)]
            score_to_plot = [scores[str(key_num)] for key_num in key_info_to_plot]
            fig, ax = plt.subplots()
            fig_name = "Key Info F1 Score for Medical Case " + str(case) + " for model " + \
                       model_configs[m]["used_model"] + "_" + model_configs[m]["method"]
            ax.bar(key_info_to_plot, score_to_plot)
            ax.set_xlabel('Key')
            ax.set_ylabel('Key F1 Score')
            ax.set_title(fig_name)
            fig.savefig('./analysis/{}.png'.format(fig_name))

    # plot scatter chart for key f1 score per case among all models
    for case in range(10):
        key_info_to_plot = case_key_map()[str(case)]
        fig, ax = plt.subplots()
        fig_name = str(case+1) + "_Key Info F1 Score for Medical Case " + str(case) + " among all models"
        for m in range(4):
            scores = key_f1_detail_list[m]
            score_to_plot = [scores[str(key_num)] for key_num in key_info_to_plot]
            label = model_configs[m]["used_model"] + "_" + model_configs[m]["method"]
            ax.scatter(score_to_plot, key_info_to_plot, label=label)
        ax.set_ylabel('Key')
        ax.set_xlabel('Key F1 Score')
        ax.set_title(fig_name)
        ax.legend()
        fig.savefig('./analysis/{}.png'.format(fig_name))
