import pandas as pd

if __name__ == '__main__':
    origin_pn_notes = pd.read_csv("DistilBert_Binary.csv", header=0)
    text = origin_pn_notes[origin_pn_notes["pn_num"] == 70792]["pn_history"].item()
    print(text[798: 806], "@", text[788: 797])
    print("here")
