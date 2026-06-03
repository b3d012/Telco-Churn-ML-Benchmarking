# Threshold Recommendation

Thresholds with precision at least 0.68 were considered acceptable.
- Best threshold for highest recall with acceptable precision: 0.25
  - Precision: 0.697
  - Recall: 0.896
  - F1: 0.784
  - Predicted positive rate: 0.603
- Best threshold for highest F1: 0.30
  - Precision: 0.699
  - Recall: 0.894
  - F1: 0.784
  - Predicted positive rate: 0.600

Business interpretation: use the recall-favoring threshold if the retention team wants to catch as many churners as possible, even if that increases false positives. Use the F1 threshold if the team wants a more balanced tradeoff between missed churners and unnecessary outreach.
The model produced only 8 distinct prediction patterns across 19 thresholds, which means the score distribution is concentrated and threshold tuning has limited effect in parts of the range.