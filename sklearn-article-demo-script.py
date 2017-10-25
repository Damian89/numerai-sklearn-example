#!/usr/bin/env python
# coding: utf8

""" Example for numer.ai competition """

import math
import os
import sys

import numpy
import pandas

__author__ = "Damian Schwyrz"
__copyright__ = "Copyright 2007, damianschwyrz.de"
__email__ = "mail@damianschwyrz.de"

# Settings
feature_cols_to_keep = []
eras_to_exclude_from_trainingset = []
kept_eras_in_trainingset = []
combine_training_and_val_data_into_one_trainingset = False
save_good_models = True
save_result_file = True
cv_number = 4
test_size_for_train_test_split = 0.25
use_adaboosting = False

# Folders
workfolder = os.getcwd() + "/"
datafolder = workfolder + "data/"
resultsfolder = workfolder + "results/"
modelfolder = workfolder + "models/"

# Perform basic file and folder checks...
if not os.path.exists(modelfolder):
    print("Creating models folder...")
    os.makedirs(modelfolder)

if not os.path.exists(resultsfolder):
    print("Creating results folder...")
    os.makedirs(resultsfolder)

# Check if both data files exist
if not os.path.exists(datafolder + "numerai_tournament_data.csv"):
    print("Sorry, numerai_tournament_data.csv not found in ./data!")
    sys.exit()

if not os.path.exists(datafolder + "numerai_training_data.csv"):
    print("Sorry, numerai_training_data.csv not found in ./data!")
    sys.exit()

# Its not recommenced to include AND exclude eras at the same time... doesn't make sense...
if len(eras_to_exclude_from_trainingset) > 0 and len(kept_eras_in_trainingset) > 0:
    print("Please don't use 'eras_to_exclude_from_trainingset' AND 'kept_eras_in_trainingset' at the same time!")
    sys.exit()

# Loading data into ram
tournament_data = pandas.read_csv(datafolder + "numerai_tournament_data.csv")
training_data = pandas.read_csv(datafolder + "numerai_training_data.csv")
validation_data = tournament_data[tournament_data.data_type == 'validation']

# Quick summary
print("Data summary after raw import:")
print("Tournament data:\t{} rows\t\t{} columns".format(tournament_data.shape[0], tournament_data.shape[1]))
print("Training data:\t\t{} rows\t\t{} columns".format(training_data.shape[0], training_data.shape[1]))
print("Validation data:\t{} rows\t\t{} columns".format(validation_data.shape[0], validation_data.shape[1]))
print()

# Columns which are not features
no_feature_cols = ['id', 'era', 'data_type', 'target']

# Get ids and features of tournament data
Ids_tournament = tournament_data.id.values
X_tournament = tournament_data.drop(no_feature_cols, axis=1)

# Sometimes you want combine trainings and validation data into one training set! ;)
# Keep in mind: its not recommended!
if combine_training_and_val_data_into_one_trainingset:
    training_data = pandas.concat([training_data, validation_data])

# Some eras are crap? Excluded them from your trainings dataset
if len(eras_to_exclude_from_trainingset) > 0:
    excluded_eras = ["era" + str(int(x)) for x in eras_to_exclude_from_trainingset]
    mask = training_data.era.isin(excluded_eras)
    training_data = training_data[~mask]

# Or the other way around: use only selected eras for training
if len(kept_eras_in_trainingset) > 0:
    eras_kept = ["era" + str(int(x)) for x in kept_eras_in_trainingset]
    mask = training_data.era.isin(eras_kept)
    training_data = training_data[mask]

# Split training data into features and targets
X_training = training_data.drop(no_feature_cols, axis=1)
y_training = training_data.target.values

# Same for validation data + get unique eras
X_validation = validation_data.drop(no_feature_cols, axis=1)
y_validation = validation_data.target.values
eras_validation = validation_data.era.unique()

# Not all features are equally important, maybe it is a good idea to include only specific features
if len(feature_cols_to_keep) > 0:
    mask = ["feature{}".format(feature_id) for feature_id in feature_cols_to_keep]
    X_tournament = X_tournament[mask]
    X_training = X_training[mask]
    X_validation = X_validation[mask]

# Maybe it is a good idea to construct you own features? Hint... think about it!
# You need to know what you are doing... this is just a plain simple example where feature 1 and 2 are multiplied and
# returned to the power 2. The returned result is used a new feature called "your_feature".
# WARNING: Don't forget to add you new features also to your validation and tournament data otherwise the fitting will fail
# feature_construct = lambda data: pow((data.feature1 * data.feature2), 2)
# X_training['your_feature'] = feature_construct(X_training)
# X_tournament['your_feature'] = feature_construct(X_tournament)
# X_validation['your_feature'] = feature_construct(X_validation)

print("Current size of training set:")
print("Rows/Data points: {}\t\tColumns/Features: {}".format(X_training.shape[0], X_training.shape[1]))
print()

# From here we need numpy arrays not the complex dataframe structures pandas is creating
X_training = X_training.values
X_tournament = X_tournament.values
X_validation = X_validation.values

# Use a simple data preprocessor
# You could also use a sklearn pipeline combining multiple preprocessors including for example principle component
# analysis or any kind of kernel approximation...
from sklearn import preprocessing, pipeline

preprocessor = pipeline.Pipeline(
    [
        ('ss', preprocessing.StandardScaler()),
        # ('pca', decomposition.PCA(n_components=15))
    ]
)

preprocessor.fit(X_training)

X_training = preprocessor.transform(X_training)
X_tournament = preprocessor.transform(X_tournament)
X_validation = preprocessor.transform(X_validation)

# Since we have validation data to check our model on, we don't need training/test split. But sometimes this
# may be a good idea to do!
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_training,
    y_training,
    test_size=test_size_for_train_test_split,
    random_state=42,
)

# Now the simple "machine learning" part... the model, we are using a simple tree classifier
# Take a look at http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
from sklearn import ensemble

model = ensemble.ExtraTreesClassifier(
    n_estimators=10,
    max_depth=3,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)  # <-- training of model happens here

# This is pretty important for our probability in the next steps. So keep this in mind!
print("Class/Target order:")
print(model.classes_)
print()

# Some estimators are able to calculate the importance of features after fitting. This is very simple way to get the
# same results while using a smaller amount of features (the more features you have, the slower training can be).
# For the next iteration you could add the generated list to the "feature_cols_to_keep"-setting above.
if not len(feature_cols_to_keep) > 0:
    print(model.feature_importances_)
    importance_average = numpy.mean(model.feature_importances_)

    print("Features with an importance above the current mean importance ({:.6f}):".format(importance_average))
    above_average_important_features = [(i + 1) for (i, importance) in enumerate(model.feature_importances_) if
                                        importance >= importance_average]
    print(above_average_important_features)
    print()

# Lets take a look at the performance using the models internal score method
score_test_data = model.score(X_test, y_test)
score_val_data = model.score(X_validation, y_validation)

print("Model: Score for {:.2f}% of training data:\t\t{:.6f}".format(
    test_size_for_train_test_split * 100,
    score_test_data
))

print("Model: Score for nmr's own validation data :\t\t{:.6f}".format(score_val_data))

if cv_number > 0:
    # Lets check out what cross_validation says:
    scores_test_data_cv = model_selection.cross_val_score(model, X_training, y_training, cv=cv_number, n_jobs=-1)
    scores_val_data_cv = model_selection.cross_val_score(model, X_validation, y_validation, cv=cv_number, n_jobs=-1)

    print("CV ({}): Score for 100% of training data:\t\t{:.6f} (+-{:.6f})".format(
        cv_number,
        test_size_for_train_test_split * 100,
        scores_test_data_cv.mean(),
        scores_test_data_cv.std() / math.sqrt(cv_number)
    ))

    print("CV ({}): Score for nmr's own validation data :\t{:.6f} (+-{:.6f})".format(
        cv_number,
        scores_val_data_cv.mean(),
        scores_val_data_cv.std() / math.sqrt(cv_number)
    ))

# What about the log loss?
from sklearn import metrics

probability_test_data = model.predict_proba(X_test)
probability_val_data = model.predict_proba(X_validation)

logloss_test_data = metrics.log_loss(y_test, probability_test_data)
logloss_val_data = metrics.log_loss(y_validation, probability_val_data)

print("Logloss for {:.2f}% training data:\t\t\t\t{:.6f}".format(
    test_size_for_train_test_split * 100,
    logloss_test_data
))

print("Logloss for validation data:\t\t\t\t\t\t{:.6f}".format(logloss_val_data))
print()

if use_adaboosting:
    print("AdaboostClassifier activated as meta-classifier, refitting...")
    ada_model = ensemble.AdaBoostClassifier(
        base_estimator=ensemble.ExtraTreesClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            n_jobs=-1,
        ),
        n_estimators=75,
        learning_rate=0.1,
        random_state=42,
    )

    ada_model.fit(X_train, y_train)

    # Lets get basic performance metrics for boosted model

    boosted_score_test_data = ada_model.score(X_test, y_test)
    boosted_score_val_data = ada_model.score(X_validation, y_validation)

    print("[Boosted] Model: Score on 33% of training data:\t\t\t{:.6f}".format(boosted_score_test_data))
    print("[Boosted] Model: Score on nmr's own validation data :\t{:.6f}".format(boosted_score_val_data))

    boosted_probability_test_data = ada_model.predict_proba(X_test)
    boosted_probability_val_data = ada_model.predict_proba(X_validation)

    boosted_logloss_test_data = metrics.log_loss(y_test, boosted_probability_test_data)
    boosted_logloss_val_data = metrics.log_loss(y_validation, boosted_probability_val_data)

    print("[Boosted] Logloss on training data:\t\t\t\t\t\t{:.6f}".format(boosted_logloss_test_data))
    print("[Boosted] Logloss on validation data:\t\t\t\t\t{:.6f}".format(boosted_logloss_val_data))
    print()

    # Lets calculate improvements
    print("Improvement:")

    improvement_test_score = (score_test_data - boosted_score_test_data) / score_test_data * 100
    print("[Testdata] Score (base classifier vs adaboosted classifier):\t\t\t{:.6f}%".format(
        improvement_test_score
    ))

    improvement_test_logloss = (logloss_test_data - boosted_logloss_test_data) / logloss_test_data * 100
    print("[Testdata] Logloss (base classifier vs adaboosted classifier):\t\t\t{:.2f}%".format(
        improvement_test_logloss
    ))

    improvement_val_score = (score_val_data - boosted_score_val_data) / score_val_data * 100
    print("[Validationdata] Score (base classifier vs adaboosted classifier):\t\t{:.6f}%".format(
        improvement_val_score
    ))

    improvement_val_logloss = (logloss_val_data - boosted_logloss_val_data) / logloss_val_data * 100
    print("[Validationdata] Logloss (base classifier vs adaboosted classifier):\t{:.2f}%".format(
        improvement_val_logloss
    ))

    # If adaboost improved our scores, lets make it the base classifier and the one to use for writing results and
    # the one to be saved for later use.
    if improvement_val_logloss > 0 and improvement_val_logloss > 0:
        model = ada_model

    print()

# Lets use our model to predict every target within the tournament data, the format numer.ai expects the results to be is:
# Id, probability (of target being 1)
probability_tournament_data = model.predict_proba(X_tournament)

# Most sklearn predict_proba methods return lists for every feature row containing the probability for every class! We
# are interested in the probability of being "1". model.classes_ shows us which element number we have to select for
# this probability. In our case it is the second element of every subarray (we start counting at 0!! ;)
# I'm pretty sure you will have to check this point on your own to get it fully!
probability_for_tournament_data_of_being_1 = probability_tournament_data[:, 1]

numer_ai_result = pandas.DataFrame(
    {
        'id': Ids_tournament,
        'probability': probability_for_tournament_data_of_being_1,
    }
)

# In the case of numerai competition also the "consistency" is important. You can find its implementation in numer.ais github:
# https://github.com/numerai/submission-criteria/blob/820d0f939ae2892f6bdeee02d855ffc0e80958de/database_manager.py#L83
# Lets copy this part!
better_than_random_era_count = 0

for era in eras_validation:
    era_data = validation_data[validation_data.era == era]
    submission_era_data = numer_ai_result[numer_ai_result.id.isin(era_data.id.values)]
    era_data = era_data.sort_values(["id"])
    submission_era_data = submission_era_data.sort_values(["id"])
    logloss = metrics.log_loss(era_data.target.values, submission_era_data.probability.values)

    if logloss < -math.log(0.5):
        better_than_random_era_count += 1

consistency = better_than_random_era_count / len(eras_validation) * 100

print("Calculated consistency: {:.2f}%".format(consistency))
print()

# You could implement also the originality and concordance metric ;)
# For now we skip those steps

# If our logloss is below ~0,631  or -math.log(0.5) AND consistency is greater than 75% we could submit our resultfile!
# Lets save/create our resultfile if this conditions are true, we are using the logloss based on numerai's validation data.
if logloss_val_data > -math.log(0.5):
    print("Sorry, logloss is only {:.6f}, thats bigger than {:.6f}".format(logloss_val_data, -math.log(0.5)))
    sys.exit()

if consistency < 75:
    print("Sorry, consistency is only {:.2f}%, thats smaller than 75%".format(consistency))
    sys.exit()

file_base_name = "sklearn-{:.6f}-{:.3f}-{:.2f}".format(logloss_val_data, score_val_data, consistency)

if save_result_file:
    print("Wrote resultfile.")
    numer_ai_result.to_csv(
        resultsfolder + file_base_name + ".csv",
        index=False
    )

# Since we found our super good model, its a good idea to save it for future use (every week starts a new challenge!).
if save_good_models:
    import pickle

    print("Saved model.")
    with open(modelfolder + file_base_name + ".model", 'wb') as model_file:
        pickle.dump(model, model_file)
