# Identify Key Information in Patient Notes from Medical Licensing Exams
This a course project of NLP course. The LLMs we used are customized based on the models provided by transformers library.

In this project, we proposed two different solutions to a Kaggle competition ([link](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/overview)), and we experimented the two solutions with DistilBERT and ELECTRA models separaterly, in which our best model achieves 84.75% overall accuracy. 

## Two Approaches
**(1) Non-Query Approach:**
![non-query](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/non-query-approach.png)

This approach takes the patient notes as input and the model predicts labels for each token as the output, where the label represents which target key information that the current token is related to. Those labels will be converted into detected character spans for each target key information in the post-processing pipeline as the final output. This approach considers all the target key information at the same time during inference, even some key information are not required to detect in the current patient note based on the medical case. For this approach, we customized a DistilBERT model and a ELECTRA-Discriminator model separately for token-level multi-classification and fine-tuned them for the experiment.

![distilMulti](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/DistilBERT-multi.png)
![electraMulti](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/ELECTRA-multi.png)

**(2) With-Query Approach:**
![with-query](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/with-query-approach.png)

This approach is different from the Non-Query approach. Besides using the patient note as input, this approach also use target key information for input as "Query", and executes a binary classification to predict whether each token refers to the target key information (the "Query") or not. We used this approach as a comparison to the Non-Query Approach, as we thought the additional information provided by the "Query" might improve the accuracy of predictions. For this approach, we customized the DistilBERT and the ELECTRA-Discriminator for token-level binary classification and fine-tuned them for our task.

![distilBinary](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/DistilBERT-Binary.png)
![electraBinary](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/ELECTRA-Binary.png)

The link to access our models: [link](https://drive.google.com/drive/folders/1mJMiY7TDsw466YCZG605hsezkEW04-my?usp=sharing)

## Analysis
We also proposed evaluation metrics to analyze the models' performance to reveal data insights. (note: For the three evaluation metrics, please check section 4.3 in ([our report](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/project%20report.pdf)).) The followings are our findings:

Comparing the micro F1 scores of the models with the baseline model (DistilBERT-Multi-classification), we found that the ELECTRA-Binary model worked best, and the ELECTRA-Multi model worked worst. While, through the comparison of two macro F1 scores of With-Query approach models (the two binary classification models) with the Non-Query approach models (the two multi-classification models), we found that With-Query approach models had higher scores on the two macro F1 metrics, which meant their performances were not affected much by poor performance on specific medical cases or key information. That indicated that they work better on the imbalanced dataset, probably benefited from the additional information provided by the "Query".

After the experiments, we took a deeper exploration to find out why the gaps between the two macro F1 scores of models were much larger than the gap between their micro F1 scores. So, we plotted the F1 scores of models on each medical case in to check if the F1-Medical scores were affected by models’ performances on a specific medical case. After that, we checked if the F1-Medical scores were affected by the performance on specific target key information. We also checked models’ quality results on those target key information for a instinctive understanding about their differences on performance.

**Fig a: model F1 scores per medical case:**
![model F1 scores per medical case](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/f1_case_overall.png)
**Fig b: F1 scores per key info on case 4:**
![F1 scores per key info on case 4](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/case4.png)
**Fig c: F1 scores per key info on case 7:**
![ F1 scores per key info on case 7](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/model-img-and-analysis/case7.png)

We found that the DistilBERT-Multi had similar performances on most medical cases as the ELECTRA-Multi, but it performed significantly better on medical case 4, which made its F1-Medical score much higher, and caused the gap between their F1-Medical scores larger than the gap between their micro F1 scores. See Figure b, when we checked the models’ performance on target key information, we found that its performance was affected by the performance on key information 403("Heavy-caffeine-use"). The prediction result of DistilBERT-Multi on this key information was more complete than the others. For instance, when predicting on patient note 42802, it identified the same character span as the ground truth, which is [’874 896’] ("Drinks 5-6 coffee cups"), while the others only identified part of the information, for instance, the ELECTRA-Multi identified [’885 891’]("coffee") and the two With-Query approach models identified [’881 896’]("5-6 coffee cups").

The same thing happened on ELECTRA-Binary model, its excellent performance on case 7 made it stand out from the other models, and that performance was affected by the performance on key information 704("Unprotected-Sex"), in which it also identified more complete information. The ground-truth on patient note 70792 is [’784 794’, ’784 786;798 810’] ("no condoms", "no contracepton"), and it identified [’784 794’, ’798 806’]("no condoms", "contrace"). While, DistilBERT-Multi model only identified [’784 794’]("no condoms") and the other two models failed to identify it.

---

For more details, please check our project report ([link](https://github.com/SemiXQ/Identify-Key-Information-in-Patient-Notes/blob/master/project%20report.pdf)). :)
