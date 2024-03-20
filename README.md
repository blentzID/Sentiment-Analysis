# 2019 Indonesian Presidential Election Tweet Sentiment Analysis


## Brief Overview

2019 Indonesian Presidential Election is a general election held in Indonesia to elect the president and vice president 
for 2019 - 2024 term. The candidates are Joko Widodo - Ma'aruf Amin and Prabowo Subianto - Sandiaga Uno.

Various text cleaning techniques and embeddings are employed to preprocess the text to be the input for the machine learning models. The machine learning models tested are mainly Random Forest model and LSTM based network. Furthermore, a benchmarking to a pre-trained BERT model is also done for performance comparison.

## Dataset

The tweets related to the candidates were collected from Twitter and they were labeled either positive, neutral or negative.  

*** Class distribution ***

| Class      | Frequency (Count) | Proportion |
| ---------- | ----------------- | ---------- |
| Positive   | 612               | 33.72%     |
| Neutral    | 607               | 33.44%     | 
| Negative   | 596               | 32.8%      | 


*** Data Exploration ***

1. Candidates names
A quick count of the candidates names mentioned in each tweet shows that both candidates are balancely mentioned for each sentiment.

| ![Candidates names distribution at each sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/president_names_exploration.PNG?raw=true) |
|:--:| 
| Candidates names distribution at each sentiment |


2. Hashtags

Similarly, counting the hashtags with respect to the sentiment of the texts shows that hashtags in the texts are inconclusive to a specific sentiment.

| ![Top 5 hashtags for each sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/hasthags_exploration.PNG?raw=true) |
|:--:| 
| Top 5 hashtags for each sentiment |


3. WordCloud

Using the WordCloud to see the most frequent words in with respect to the candidate and its sentiment.

| ![Wordcloud for Jokowi with positive sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/wordcloud_jokowi_positif.png?raw=true) |
|:--:| 
| Wordcloud for Jokowi with positive sentiment |

| ![Wordcloud for Prabowo with positive sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/wordcloud_prabowo_positif.png?raw=true) |
|:--:| 
| Wordcloud for Prabowo with positive sentiment |


Findings:
- There are a lot of stopwords, e.g. conjucntions in the text
- Some keywords are mentioned in the texts for both candidates, e.g. 'ekonomi' appears in both candidates wordcloud
- Some keywords are specific for each candidate, e.g. 'gaji' mainly appears for Prabowo, whereas 'harga' mainly appears for Jokowi


## Text Preprocessing

| ![Text cleaning workflow](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/text_cleaning.PNG) |
|:--:| 
| Text cleaning workflow |


The steps in text cleaning:
1. Remove any url appears in the text
2. Remove any hashtag in the text
3. Normalize the slang words used in the text by referring to a slang words vocab (the slang word vocab is taken from [here](https://github.com/nasalsabila/kamus-alay/blob/master/colloquial-indonesian-lexicon.csv))
4. Remove stopwords by referring to a customized slang words vocab
4.a. Take a standard stopwords vocab from nlp_id library
4.b. Run word level postagging to tag each stopword
4.c. Take all stopwords which are not NEG (negation), JJ (adjective), VB (verb), FW (foreign words), and NUM (number) as the custom stopwords vocab
5. Run phrase level postagging for the preprocessed text
5.a. If the phrase is a number, normalize it to 'NUM'
5.b. If the phares is a verb, lemmatize it


*** Data Splitting ***  
Test size is 20% of the dataset, and the sampling strategy is stratified sampling.


*** Label Encoding ***  
The original labels are string, so a label encoding is done to convert the labels into numbers


## Embedding and Modeling

Various word embeddings are combined with Random Forest and LSTM-based network.   
The metrics to evaluate the models' performance is the accuracy on the validation dataset.  

The result is as following:

| Class         | TF-IDF | Word2Vec | Embedding Layer Tensorflow |
| ------------- | ------ | -------- | -------------------------- |
| Random Forest | 61.15% |  61.4%   |             -              |
| LSTM          | 57.38% |  58.9%   |          58.41%            |


Findings:
1. All experiments suffer from the overfitting, indicated by good training performance, where validation performance does not improve
   
| ![Learning Curve for LSTM + Embedding Layer](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/lstm_embedding_learning_curve.png?raw=true) |
|:--:| 
| Learning Curve for LSTM + Embedding Layer |

3. In general, using Random Forest is better than using LSTM for this use case


*** Benchmarking ***

A pre-trained BERT model by mdhugol/indonesia-bert-sentiment-classification (https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification) is employed to predict the sentiment on the test set.

The pre-trained BERT model's validation accuracy is 63.63%.  
This shows that the performance of the best model created, i.e. the Random Forest with Word2Vec, is comparable to the pre-trained BERT model for this dataset.

On top of the validation accuracy value, the confusion matrix on the test set by the BERT and Random Forest + Word2Vec are similar, where the models are able to predict the negative sentiment better than the neutral and positive class.
| ![Confusion Matrix for BERT (left) and Random Forest + Word2Vec (right)](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/cm_bert_rfw2v.PNG) |
|:--:| 
| Confusion Matrix for BERT (left) and Random Forest + Word2Vec (right) |

## Conclusion
1. For this specific use case, the best model is random forest with word2vec word embedding, which produces validation accuracy of 61.7%
2. Benchmarking to pretrained models is done, and the best pre-trained model is the BERT model with validation accuracy of 63.63%
3. The results show that the models suffer from overfitting

## Improvement
1. Improving the cleaning process
2. In this case, using deep learning models seem to be not better than machine learning models, so, focus on using machine learning models
3. Fix the overfitting issue
4. Run error analysis on the result, and start fixing from the most misclassified class

