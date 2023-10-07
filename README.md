# Deep-Learning-Arxiv-Paper-Recommender
The final project of my deep learning class from my Master's in Analytics at Georgia Tech

### Summary
Current research paper recommendation systems like
Google Scholar and Microsoft Academic rely heavily on
citation counts to serve their recommendations. Scholas-
tic Index is a novel paper recommending system that serves
recommendations based on content similarity instead. Sev-
eral models, including term frequency-inverse document
frequency (tf-idf), transformer, and long short-term mem-
ory (lstm) are used to generate embeddings for an abstract,
which are compared by the cosine similarity measure. This
method has advantages to citation count because it may find
a paper with very relevant content but no citations. Experi-
mental results demonstrate the method is sound and delivers
relevant papers.

### Approach
This ArXiv paper recommender pro-
vides users with the most similar papers compared to a pa-
per of interest. The approach involves loading and cleaning
the abstracts for a specific ArXiv category, experimenting
with using several deep learning and NLP models to cre-
ate embedding representations for each of the abstracts, and
then ranking the similarity of documents in comparison to
each other using a cosine similarity matrix. For a given pa-
per, the application provides the top 20 most similar papers,
which can help the user find additional papers related to the
topic and their interests.

### Data Processing
The dataset is from Kaggle and provided by [ArXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
It contains 1.7M entries and is a JSON document. Each
entry contains important information such as author, title,
category, abstract, and comments. The following
categories are used: cs.LG (Machine Learning), cs.NE (Neural and
Evolutionary Computing) and cs.LG and cs.NE with a to-
tal of 112,608 entries. The pandas data frame consists of
112,608 papers with 5 columns (id, abstract, title, authors,
and categories). Since there are processing time constraints, I selected 5,000 samples and
preprocessed the abstracts with lemmatization and the re-
moval of capitalization, numerical factors, punctuation, and
stop words.
Once the data was ready, I experimented with several
models outlined below. The inputs and outputs into all the
models are consistent, where the input is the abstracts of
the ArXiv research papers, and the output is embeddings
for each of the abstracts.

### TF-IDF model
For TFIDF (Term Frequency-Inverse Document Fre-
quency), I used the ‘gensim’ package in Python to first create a dictionary where each word in the abstract was tok-
enized and stored as a key. I then converted the dictionar-
ies of words to a Bag of Words representation, a frequency
count of each word in each abstract. The Bag of Words Cor-
pus was used as the input to the TFIDF model. The TFIDF
model first computes the term frequency (TF) by dividing
the count of words in the document by the word count in
the entire corpus. This term frequency is then multiplied by
the global component IDF, the log of the ratio of the num-
ber of abstracts divided by the number of abstracts with the
word. This metric effectively computes a normalized bag
of words vector that adjusts the scores based on how com-
mon a word is across the corpus of abstracts. The output of
the TFIDF model are embeddings that are used to create the
cosine similarity matrix betIen each abstract.

### Transformer
For the transformer model, I used the ‘sentence-
transformers’ python library for the sentence BERT model
and chose the ”all-distilroberta-v1” pretrained transformer
model to generate embeddings for the ArXiv research ab-
stracts. I chose this model because
of the high-performance scores in generating sentence em-
beddings. The transformer mapped each abstract to a 768-
dimensional dense vector space that was trained on a 1B
sentence pair dataset with a contrastive learning objective,
where the model was given one of the sentences from the
pair and predicted which sentence was the pairing given
a random sample of sentences. Similar to the other mod-
els, the preprocessed abstracts are used as the input to the
model and the output is sentence embeddings, which are
used to create a cosine similarity matrix that can generate
recommendations.

### Evaluation
Once the embeddings are generated, I compared them
using a cosine similarity metric in the form of a 5,000 x
5,000 matrix where the value represents pairwise similarity
betIen the paper represented by the row and the paper rep-
resented by the column. Diagonal entries have a value of 1
since papers are fully similar to themselves. A higher value
means that the paper is more similar to the parent paper than
another paper with a loIr value. For our experiments, I
selected “Reinforcement Learning with Augmented Data”
as the parent paper to generate recommendations against
and to compare models.
To measure success, I retrieved the top 20 most similar
embeddings and their respective abstracts as shown in fig-
ures 9 and 10. To evaluate the quality of a model’s recom-
mendation, a subject matter expert (us) received a random-
ized list of the top 20 and made their own top 10 ranking.
These two lists are compared to see how much overlap
there was in papers and ranking. If there was significant
overlap, then this model did a good job of serving recom-
mendations in order of relevance. To compare the quality
of recommendations betIen models, I assigned a rele-
vancy score betIen 1 and 5 to each recommendation and
compared the average relevance for a model.

### Analysis of Results
The figure below shows the percentage of ranking overlap for TF-
IDF and RoBERTa amongst all reviewers. 
![Overlap!](/img/f8overlap.png)

TF-IDF has sig-
nificantly better performance than RoBERTa with an aver-
age overlap of approximately 73%, whereas RoBERTa has
an average overlap of 30%. This makes sense because TF-
IDF matches for relevant words and reviewers ranked pa-
pers highly that had key phrases like “reinforcement learn-
ing.” 

The figure below shows the distribution of relevancy scores for
both models amongst the reviewers. 
![Overlap!](/img/f6distrelevancy.png)


The Transformer has a
significantly larger share of low scores, whereas the TF-IDF
had many papers score highly. 

The figure below shows the average
relevance of the models
![Overlap!](/img/f7avgrelevance.png)

It’s clear that TF-IDF gives many
more relevant results than RoBERTa. This makes sense be-
cause the limited dataset gives a bias towards TF-IDF since
it’s easier to match keywords than to make connections,
which would require much more data for the transformer.
To demonstrate this, the TF-IDF results have the word “re-
inforcement” appear 43 times compared with 28 times for
the RoBERTa results.
The graphs below show the t-SNE (t-Distributed Stochas-
tic Neighbor Embedding) plots for the two models. 

![Overlap!](/img/tsneroberta.png)
![Overlap!](/img/tsnetfidf.png)

t-SNE
reduces the embeddings to two dimensions by calculating
a joint Gaussian probability distribution betIen each em-
bedding in the original high-dimensional space. Then it
creates a probability distribution betIen embeddings in
a two-dimensional space using cosine similarity and min-
imizes the KL divergence betIen the probability distribu-
tions. Each point in the plots represents an abstract
embedding reduced to two dimensions. The red points high-
light the top 10 most similar documents to the first entry of
our ArXiv paper dataset. The most similar documents are
very close to each other, displaying that the t-SNE plots vi-
sualize the results of the cosine similarity matrix well. Still,
it will not be exact since they are different methods to rep-
resent similarity.
Qualitatively, the TF-IDF recommendation results have a
sense of relevancy because most of the abstracts have sim-
ilar keywords to the parent paper since the model creates
embeddings based on the amount of “rare words” shared be-
tIen documents. For example, the terms “Reinforcement
Learning,” “algorithm,” “efficient,” and “performance” are
shared across many of the recommendations, and it turns
out this is an excellent way to discover related papers.
RoBERTa, however, gives many results that have a sem-
blance of relevancy such as general machine learning pa-
pers, but because it doesn’t have the focus on keywords, the
relevancy suffers if more targeted results are desired.
