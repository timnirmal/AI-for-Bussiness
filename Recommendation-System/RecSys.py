# About datasets
# https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop?select=shared_articles.csv

############ shared_articles.csv
# Contains information about the articles shared in the platform. Each article has its sharing
# date (timestamp), the original url, title, content in plain text, the article' lang (Portuguese: pt or English: en)
# and information about the user who shared the article (author).
#
# There are two possible event types at a given timestamp:
# CONTENT SHARED: The article was shared in the platform and is available for users.
# CONTENT REMOVED: The article was removed from the platform and not available for further recommendation.

# For the sake of simplicity, we only consider here the "CONTENT SHARED" event type


############ users_interactions.csv
# Contains logs of user interactions on shared articles. It can be joined to articles_shared.csv by contentId column.
#
# The eventType values are:
# VIEW: The user has opened the article.
# LIKE: The user has liked the article.
# COMMENT CREATED: The user created a comment in the article.
# FOLLOW: The user chose to be notified on any new comment in the article.
# BOOKMARK: The user has bookmarked the article for easy return in the future.

import math
import random

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

articles_df = pd.read_csv('data/shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
print(articles_df.head(5))

interactions_df = pd.read_csv('data/users_interactions.csv')
print(interactions_df.head(10))

########################################## Data Munging ##########################################

event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0,
    'BOOKMARK': 2.5,
    'FOLLOW': 3.0,
    'COMMENT CREATED': 4.0,
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])
print(interactions_df.head(10))

########################################## Solving Cold Start ##########################################

# Recommender systems have a problem known as user cold-start, in which is hard do provide personalized
# recommendations for users with none or a very few number of consumed items, due to the lack of information to model
# their preferences.
# For this reason, we are keeping in the dataset only users with at leas 5 interactions.

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
    ['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,
                                                            how='right',
                                                            left_on='personId',
                                                            right_on='personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
print(interactions_from_selected_users_df.head(10))
print(users_with_enough_interactions_df.head(10))


# In Deskdrop, users are allowed to view an article many times, and interact with them in different ways (eg. like or
# comment). Thus, to model the user interest on a given article, we aggregate all the interactions the user has
# performed in an item by a weighted sum of interaction type strength and apply a log transformation to smooth the
# distribution.

def smooth_user_preference(x):
    return math.log(1 + x, 2)


interactions_full_df = interactions_from_selected_users_df \
    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
    .apply(smooth_user_preference).reset_index()

print('# of unique user/item interactions: %d' % len(interactions_full_df))
print(interactions_full_df.head(10))

# Evaluation
# Evaluation is important for machine learning projects, because it allows to compare objectivelly different algorithms and hyperparameter choices for models.
# One key aspect of evaluation is to ensure that the trained model generalizes for data it was not trained on, using Cross-validation techniques. We are using here a simple cross-validation approach named holdout, in which a random data sample (20% in this case) are kept aside in the training process, and exclusively used for evaluation. All evaluation metrics reported here are computed using the test set.
#
# Ps. A more robust evaluation approach could be to split train and test sets by a reference date, where the train set is composed by all interactions before that date, and the test set are interactions after that date. For the sake of simplicity, we chose the first random approach for this notebook, but you may want to try the second approach to better simulate how the recsys would perform in production predicting "future" users interactions.

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                               stratify=interactions_full_df['personId'],
                                                               test_size=0.20,
                                                               random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

"""In Recommender Systems, there are a set metrics commonly used for evaluation. We chose to work with Top-N accuracy 
metrics, which evaluates the accuracy of the top recommendations provided to a user, comparing to the items the user 
has actually interacted in test set. This evaluation method works as follows: 

For each user For each item the user has interacted in test set Sample 100 other items the user has never interacted. 
Ps. Here we naively assume those non interacted items are not relevant to the user, which might not be true, 
as the user may simply not be aware of those not interacted items. But let's keep this assumption. Ask the 
recommender model to produce a ranked list of recommended items, from a set composed one interacted item and the 100 
non-interacted ("non-relevant!) items Compute the Top-N accuracy metrics for this user and interacted item from the 
recommendations ranked list Aggregate the global Top-N accuracy metrics 

The Top-N accuracy metric choosen was Recall@N which evaluates whether the interacted item is among the top N items (
hit) in the ranked list of 101 recommendations for a user. Ps. Other popular ranking metrics are NDCG@N and MAP@N, 
whose score calculation takes into account the position of the relevant item in the ranked list (max. value if 
relevant item is in the first position). You can find a reference to implement this metrics in this post. 


"""

########################################## Evaluation ##########################################


# Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100


class ModelEvaluator:

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id,
                                               items_to_ignore=get_items_interacted(person_id,
                                                                                    interactions_train_indexed_df),
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id,
                                                                               sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                                               seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['contentId'].values
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df


model_evaluator = ModelEvaluator()


########################################## Popularity Model ##############################################

"""

A common (and usually hard-to-beat) baseline approach is the Popularity model. This model is not actually 
personalized - it simply recommends to a user the most popular items that the user has not previously consumed. As 
the popularity accounts for the "wisdom of the crowds", it usually provides good recommendations, generally 
interesting for most people. Ps. The main objective of a recommender system is to leverage the long-tail items to the 
users with very specific interests, which goes far beyond this simple technique. 

"""

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
print(item_popularity_df.head(10))


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
            .sort_values('eventStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['eventStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


popularity_model = PopularityRecommender(item_popularity_df, articles_df)

"""

Here we perform the evaluation of the Popularity model, according to the method described above. It achieved the 
Recall@5 of 0.2417, which means that about 24% of interacted items in test set were ranked by Popularity model among 
the top-5 items (from lists with 100 random items). And Recall@10 was even higher (37%), as expected. It might be 
surprising to you that usually Popularity models could perform so well! 

"""

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s\n' % pop_global_metrics)
print(pop_detailed_results_df.head(10))



########################################## Content Based Filtering Model ##############################################

"""

Content-based filtering approaches leverage description or attributes from items the user has interacted to recommend 
similar items. It depends only on the user previous choices, making this method robust to avoid the cold-start 
problem. For textual items, like articles, news and books, it is simple to use the raw text to build item profiles 
and user profiles. Here we are using a very popular technique in information retrieval (search engines) named TF-IDF. 
This technique converts unstructured text into a vector structure, where each word is represented by a position in 
the vector, and the value measures how relevant a given word is for an article. As all items will be represented in 
the same Vector Space Model, it is to compute similarity between articles. See this presentation (from slide 30) for 
more information on TF-IDF and Cosine similarity. 

"""

#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
print(tfidf_matrix)


def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile


def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles


def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])

    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
    # Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
        user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm


def build_users_profiles():
    interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
        .isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles


user_profiles = build_users_profiles()

print(len(user_profiles))

print(user_profiles)

# Let's take a look in the profile. It is a unit vector of 5000 length. The value in each position represents how relevant is a token (unigram or bigram) for me.

myprofile = user_profiles[-1479311724257856983]
print(myprofile.shape)
pd.DataFrame(sorted(zip(tfidf_feature_names,
                        user_profiles[-1479311724257856983].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance'])


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


content_based_recommender_model = ContentBasedRecommender(articles_df)


print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
print(cb_detailed_results_df.head(10))



########################################## Collaborative Filtering Model ##############################################

"""

Collaborative Filtering model¶
Collaborative Filtering (CF) has two main implementation strategies:

Memory-based: This approach uses the memory of previous users interactions to compute users similarities based on 
items they've interacted (user-based approach) or compute items similarities based on the users that have interacted 
with them (item-based approach). 

A typical example of this approach is User Neighbourhood-based CF, in which the top-N similar users (usually computed 
using Pearson correlation) for a user are selected and used to recommend items those similar users liked, 
but the current user have not interacted yet. This approach is very simple to implement, but usually do not scale 
well for many users. A nice Python implementation of this approach in available in Crab. 

Model-based: This approach, models are developed using different machine learning algorithms to recommend items to 
users. There are many model-based CF algorithms, like neural networks, bayesian networks, clustering models, 
and latent factor models such as Singular Value Decomposition (SVD) and, probabilistic latent semantic analysis. 

"""

################## Matrix Factorization
"""
Matrix Factorization
Latent factor models compress user-item matrix into a low-dimensional representation in terms of latent factors. One advantage of using this approach is that instead of having a high dimensional matrix containing abundant number of missing values we will be dealing with a much smaller matrix in lower-dimensional space.
A reduced presentation could be utilized for either user-based or item-based neighborhood algorithms that are presented in the previous section. There are several advantages with this paradigm. It handles the sparsity of the original matrix better than memory based ones. Also comparing similarity on the resulting matrix is much more scalable especially in dealing with large sparse datasets.

Here we a use popular latent factor model named Singular Value Decomposition (SVD). There are other matrix factorization frameworks more specific to CF you might try, like surprise, mrec or python-recsys. We chose a SciPy implemenation of SVD because it is available on Kaggle kernels. P.s. See an example of SVD on a movies dataset in this blog post.

An important decision is the number of factors to factor the user-item matrix. The higher the number of factors, the more precise is the factorization in the original matrix reconstructions. Therefore, if the model is allowed to memorize too much details of the original matrix, it may not generalize well for data it was not trained on. Reducing the number of factors increases the model generalization.

"""

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId',
                                                          columns='contentId',
                                                          values='eventStrength').fillna(0)

print(users_items_pivot_matrix_df.head(10))

users_items_pivot_matrix = users_items_pivot_matrix_df.values
print(users_items_pivot_matrix[:10])

users_ids = list(users_items_pivot_matrix_df.index)
print(users_ids[:10])

users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
print(users_items_pivot_sparse_matrix)

#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
#U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)

print(U.shape)
print(Vt.shape)

sigma = np.diag(sigma)
print(sigma.shape)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
print(all_user_predicted_ratings)

all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
print(cf_preds_df.head(10))

print(len(cf_preds_df.columns))


class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

# Evaluating the Collaborative Filtering model (SVD matrix factorization), we observe that we got Recall@5 (33%) and
# Recall@10 (46%) values, much higher than Popularity model and Content-Based model.

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
print(cf_detailed_results_df.head(10))



########################################## Hybrid Model ##############################################


# What if we combine Collaborative Filtering and Content-Based Filtering approaches?
# Would that provide us with more accurate recommendations?

# In fact, hybrid methods have performed better than individual approaches in many studies and have being extensively
# used by researchers and practioners.

# Let's build a simple hybridization method, as an ensemble that takes the weighted average of the normalized CF
# scores with the Content-Based scores, and ranking by resulting score. In this case, as the CF model is much more
# accurate than the CB model, the weights for the CF and CB models are 100.0 and 1.0, respectivelly.

class HybridRecommender:
    MODEL_NAME = 'Hybrid'

    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'recStrength': 'recStrengthCB'})

        # Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'recStrength': 'recStrengthCF'})

        # Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how='outer',
                                   left_on='contentId',
                                   right_on='contentId').fillna(0.0)

        # Computing a hybrid recommendation score based on CF and CB scores
        # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) \
                                       + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')[
                ['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df


hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
print(hybrid_detailed_results_df.head(10))


########################################## Comparing the methods ##############################################

global_metrics_df = pd.DataFrame([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics]) \
                        .set_index('modelName')
print(global_metrics_df)


ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

######################################### Testing #########################################

def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how = 'left',
                                                      left_on = 'contentId',
                                                      right_on = 'contentId') \
                          .sort_values('eventStrength', ascending = False)[['eventStrength',
                                                                          'contentId',
                                                                          'title', 'url', 'lang']]

print(inspect_interactions(-1479311724257856983, test_set=False).head(20))

print(hybrid_recommender_model.recommend_items(-1479311724257856983, topn=20, verbose=True))

# Sample out
# 	recStrengthHybrid	contentId	title	url	lang
# 0	25.436876	3269302169678465882	The barbell effect of machine learning.	http://techcrunch.com/2016/06/02/the-barbell-e...	en
# 1	25.369932	-8085935119790093311	Graph Capabilities with the Elastic Stack	https://www.elastic.co/webinars/sneak-peek-of-...	en