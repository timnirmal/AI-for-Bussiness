# Recommendation System

There are mainly 3 families of recommendation systems:

1. **Collabarative Filtering**:

This method makes automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on a set of items, A is more likely to have B's opinion for a given item than that of a randomly chosen person.

2. **Content-Based Filtering**:

This method uses only information about the description and attributes of the items users has previously consumed to model user's preferences. In other words, these algorithms try to recommend items that are similar to those that a user liked in the past (or is examining in the present). In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended.

3. **Hybrid**:

Recent research has demonstrated that a hybrid approach, combining collaborative filtering and content-based filtering could be more effective than pure approaches in some cases. These methods can also be used to overcome some of the common problems in recommender systems such as cold start and the sparsity problem.


We have implemented all three methods in our system and Output is taken from hybrid method.
Giving 100:1 ratio of weight to collaborative filtering and content-based filtering.


links:   

[Dataset](https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop?select=shared_articles.csv)   
[Article](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101/notebook)
