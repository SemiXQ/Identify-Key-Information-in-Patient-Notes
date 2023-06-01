from ModelsFuncs import BertClassifierBinary, BertClassifierMulti, ElectraMulti, ElectraBinary
from ModelsFuncs import preprocess_multi, preprocess_binary, train_pipeline, val_test_pipeline, test_macros, eval_pipeline, form_out_character_span
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


BATCH_SIZE = 16
DEVICE = "cuda"

LEARNING_RATES = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
EPOCHS = [5, 10, 15, 20]


def preprocess(method, tokenizer, batch_size):
    if method == "Multi":
        return preprocess_multi(tokenizer=tokenizer, batch_size=batch_size)
    else:
        return preprocess_binary(tokenizer=tokenizer, batch_size=batch_size)


if __name__ == '__main__':
    TRAIN = False  # True
    EVAL_QUANTITY = False  # True
    EPOCH = 20
    learning_rate = 3e-5

    # model = BertClassifierMulti()
    # used_model = "DistilBert"
    # method = "Multi"
    # num_classes = 144

    # model = ElectraMulti()
    # used_model = "ELECTRA"
    # method = "Multi"
    # num_classes = 144

    # model = BertClassifierBinary()
    # used_model = "DistilBert"
    # method = "Binary"
    # num_classes = 2

    model = ElectraBinary()
    used_model = "ELECTRA"
    method = "Binary"
    num_classes = 2

    tokenizer = model.tokenizer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    model.to(DEVICE)

    train_loader, val_loader, test_loader = preprocess(method=method, tokenizer=tokenizer, batch_size=BATCH_SIZE)

    train_loss_list = []
    val_loss_list = []
    train_f1_list = []
    val_f1_list = []

    model_name_final = used_model + "_" + method + "_nbme_bert_v2.pth"

    if TRAIN:
        print(used_model, " epoch: ", EPOCH, "LR: ", learning_rate)
        for epoch in range(EPOCH):
            print("Epoch ", epoch, " started")
            start = time.time()
            train_loss, train_f1_score = train_pipeline(model, optimizer, criterion, DEVICE, train_loader, num_classes)
            train_loss_list.append(train_loss)
            train_f1_list.append(train_f1_score)
            train_end = time.time()
            print("Training: Epoch ", epoch, " loss ", train_loss, " f1 ", train_f1_score, " time:", train_end-start)
            model_name = str(epoch) + "_" + used_model + "_" + method + '_nbme_bert_v2.pth'
            torch.save(model.state_dict(), model_name)
            val_loss, val_f1_score = val_test_pipeline(model, criterion, DEVICE, val_loader, num_classes)
            val_loss_list.append(val_loss)
            val_f1_list.append(val_f1_score)
            val_end = time.time()
            print("Validation: Epoch ", epoch, " loss ", val_loss, " f1 ", val_f1_score, " time:", val_end-train_end)

        torch.save(model.state_dict(), model_name_final)
    else:
        weights = torch.load(model_name_final)
        model.load_state_dict(weights)

    # test
    test_start = time.time()
    _, test_f1_score = val_test_pipeline(model, criterion, DEVICE, test_loader, num_classes)
    test_end = time.time()
    print("Test: f1 ", test_f1_score, " time:", test_end - test_start)

    case_f1_macro, keyinfo_lst = test_macros(model, DEVICE, test_loader, num_classes, type=method)
    print(case_f1_macro)
    print("case_f1: ", sum(case_f1_macro.values()) / 10)
    print(keyinfo_lst)
    print("key_f1: ", sum(keyinfo_lst.values()) / 143)

    if TRAIN:
        # plot loss and f1 score
        fig, ax = plt.subplots()
        ax.plot([epoch for epoch in range(EPOCH)], train_loss_list, label="train")  # plot train loss
        ax.plot([epoch for epoch in range(EPOCH)], val_loss_list, label="val")  # plot val loss
        ax.legend()

        fig.savefig('{}.png'.format(used_model + "_" + method + "_" + str(EPOCH) + "_" + str(learning_rate) + '_loss'))

        fig2, ax2 = plt.subplots()
        ax2.plot([epoch for epoch in range(EPOCH)], train_f1_list, label="train")  # plot train f1 score
        ax2.plot([epoch for epoch in range(EPOCH)], val_f1_list, label="val")  # plot val f1 score
        ax2.legend()

        fig2.savefig('{}.png'.format(used_model + "_" + method + "_" + str(EPOCH) + "_" + str(learning_rate) + '_f1'))
    
    if not TRAIN and EVAL_QUANTITY:
        detected_character_spans, key_info_nums, pn_nums = eval_pipeline(model, DEVICE, test_loader, method)
        df_dic = {
            "pn_num": pn_nums,
            "key_info_num": key_info_nums,
            "character_span": form_out_character_span(detected_character_spans)
        }

        df_postprocessed = pd.DataFrame(df_dic)
        file_name = used_model + "_" + method
        df_postprocessed.to_csv("{}.csv".format(file_name))




