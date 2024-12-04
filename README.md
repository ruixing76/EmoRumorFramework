# EmoRumorFramework
Code for the paper An Emotion Analysis Framework for Rumours in Online Social Media

# Dataset
Please download and extract data to raw_data directory, the data dir should look like the following:
```
- raw_data
	- pheme
		- pheme_single_aff.csv
	- twitter15
		- twitter15_single_aff.csv
    - twitter16
		- twitter16_single_aff.csv
	- fig # directory for plotted figure
```

All data are organized in a unified dataframe format. The specific columns may vary, but the commonly useful attributes are as follows:

```
id: UID of the tweet
text: tweet text
rumour_type: rumours or non-rumours
emollm_emotion: a string represents multi-label emotion detected by EmoLLM, different emotions are separated by comma
emollm_sentiment: sentiment label detected by EmoLLM
structure: json structure of the conversation, only available for root tweets. For comments, this column will be 'reaction'.
factuality: true, false or unverified
```