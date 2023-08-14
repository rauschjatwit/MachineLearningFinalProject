import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    print('\n\n\nNaive Bayes:\n\n\n')


    df = pd.read_csv('ParisHousingClass99.88.csv', header=0, sep=',')

    print(df.head)

    X = df.drop(['category'], axis=1)
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print(X_train.shape)
    print(X_test.shape)
    print(X_train.isnull().sum())
    print(X_test.isnull().sum())

    cols = X_train.columns
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    print(X_train.head())

    #
    # Check Accuracy Score
    #
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    print(y_pred)
    print('Model Accuracy Score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    #
    # Compare the train-set and test-set accuracy
    #
    y_pred_train = gnb.predict(X_train)
    print(y_pred_train)
    print('Training-set Accuracy Score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))

    #
    # Check for Over-fitting and Under-fitting
    #
    print('Training set Score: {:.4f}'.format(gnb.score(X_train, y_train)))
    print('Test set Score: {:.4f}'.format(gnb.score(X_test, y_test)))

    #
    # Compare model accuracy with null accuracy
    #
    print(y_test.value_counts())

    null_accuracy = (2615 / (2615 + 385))
    print('Null Accuracy Score: {0:0.4f}'.format(null_accuracy))

    #
    # Confusion Matrix
    #
    cm = confusion_matrix(y_test, y_pred)

    print('Confusion Matrix:\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlOrBr')
    plt.show()

    #
    # Classification Metrics
    #
    print(classification_report(y_test, y_pred))
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    print('Classification Accuracy: {0:0.4f}'.format(classification_accuracy))

    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Classification Error: {0:0.4}'.format(classification_error))

    #
    # Precision
    #
    precision = TP / float(TP + FP)
    print('Precision: {0:0.4}'.format(precision))

    #
    # Recall
    #
    recall = TP / float(TP + FN)
    print('Recall or Sensetivity: {0:0.4}'.format(recall))

    #
    # True Positive Rate
    #
    true_positive_rate = TP / float(TP + FN)
    print('True Positive Rate: {0:0.4}'.format(true_positive_rate))

    #
    # False Positive Rate
    #
    false_positive_rate = FP / float(FP + TN)
    print('False Positive Rate: {0:0.4}'.format(false_positive_rate))

    #
    # Specificity
    #
    specificity = TN / float(TN + FP)
    print('Specificity: {0:0.4}'.format(specificity))

    #
    # Probabilities
    #
    y_pred_prob = gnb.predict_proba(X_test)[0:10]
    print(y_pred_prob)

    y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of basic', 'Prob of luxury'])
    print(y_pred_prob_df)

    print(gnb.predict_proba(X_test)[0:10, 1])
    y_pred1 = gnb.predict_proba(X_test)[:, 1]

    plt.rcParams['font.size'] = 12
    plt.hist(y_pred1, bins=10)
    plt.title('Histogram of Predicted Probabilities of Luxury')
    plt.xlim(0,1)
    plt.xlabel('Predicted Probabilities of Luxury')
    plt.ylabel('Frequency')
    plt.show()

    #
    # ROC - AUC
    #
    fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label='luxury')

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Quality')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()

    ROC_AUC = roc_auc_score(y_test, y_pred1)
    print('ROC AUC: {:.4}'.format(ROC_AUC))

    Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print('Cross Validated ROC_AUC: {:.4f}'.format(Cross_validated_ROC_AUC))

    #
    # K-Fold Cross Validation
    #
    scores = cross_val_score(gnb, X_train, y_train, cv=10, scoring='accuracy')
    print('Cross Validation Scores: {}'.format(scores))
    print('Average Cross-Validation Score: {:.4}'.format(scores.mean()))

    print('\n\n\n Logistic Regression:\n\n\n')

    print(df.head)

    X = df.drop(['category'], axis=1)
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
    print(X_train[numerical].describe())

    cols = X_train.columns
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    print(X_train.describe())

    logreg = LogisticRegression(solver='liblinear', random_state=0)
    logreg.fit(X_train, y_train)

    y_pred_test = logreg.predict(X_test)
    print(y_pred_test)

    #
    # Accuracy Score
    #
    print('Model Accuracy Score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

    y_pred_train = logreg.predict(X_train)
    print(y_pred_train)

    print('Training-Set Accuracy Score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))

    #
    # Check for Over-fitting and Under-fitting
    #
    print('Training Set Score: {:.4}'.format(logreg.score(X_train, y_train)))
    print('Test Set Score: {:.4}'.format(logreg.score(X_test, y_test)))

    #
    # Logistic Regression with C = 100
    #
    logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)
    logreg100.fit(X_train, y_train)
    print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

    #
    # Logistic Regression with C = .01
    #
    logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)
    logreg001.fit(X_train, y_train)
    print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))

    #
    # Compare Model Accuracy with Null Accuracy
    #
    print(y_test.value_counts())

    null_accuracy = (2213 / (2213 + 787))
    print('Null Accuracy Score: {0:0.4}'.format(null_accuracy))

    #
    # Confusion Matrix
    #
    cm = confusion_matrix(y_test, y_pred_test)
    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

    #
    # Classification Report
    #
    print(classification_report(y_test, y_pred_test))
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    print('Classification Accuracy: {0:0.4f}'.format(classification_accuracy))

    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Classification Error: {0:0.4f}'.format(classification_error))

    #
    # Precision
    #
    precision = TP / float(TP + FP)
    print('Precision: {0:0.4f}'.format(precision))

    #
    # Recall
    #
    recall = TP / float(TP + FN)
    print('Recall: {0:0.4f}'.format(recall))

    #
    # True Positive Rate
    #
    true_positive_rate = TP / float(TP + FN)
    print('True Positive Rate: {0:0.4f}'.format(true_positive_rate))

    #
    # False Positive Rate
    #
    false_positive_rate = FP / float(FP + TN)
    print('False Positive Rate: {0:0.4f}'.format(false_positive_rate))

    #
    # Specificity
    #
    specificity = TN / (TN + FP)
    print('Specificity: {0:0.4f}'.format(specificity))

    y_pred_prob = logreg.predict_proba(X_test)[0:10]
    print(y_pred_prob)

    y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of Basic', 'Prob of Luxury'])
    print(y_pred_prob_df)

    print(logreg.predict_proba(X_test)[0:10, 1])
    y_pred1 = logreg.predict_proba(X_test)[:, 1]

    plt.rcParams['font.size'] = 12
    plt.hist(y_pred1, bins=10)
    plt.title('Histogram of Predicted Probabilities of Basic')
    plt.xlim(0, 1)
    plt.xlabel('Predicted Probabilities of Basic')
    plt.ylabel('Frequency')
    plt.show()

    #
    # ROC Curve
    #
    fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label='Yes')
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.rcParams['font.size'] = 12
    plt.title('ROC Curve for Paris Apartment Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()

    #
    # ROC-AUC
    #
    ROC_AUC = roc_auc_score(y_test, y_pred1)
    print('ROC AUC: {0:0.4f}'.format(ROC_AUC))

    #
    # K-Fold Cross Validation
    #
    scores = cross_val_score(logreg, X_train, y_train, cv=5, scoring='accuracy')
    print('Cross-Validation Scores {}'.format(scores))

    print('Average Cross-Validation Score: {0:0.4f}'.format(scores.mean()))

    #
    # Hyper-Parameter Optimization
    #
    parameters = [{'penalty': ['l1', 'l2']},
                  {'C':[1, 10, 100, 1000]}]
    grid_search = GridSearchCV(estimator=logreg,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=4,
                               verbose=0)
    grid_search.fit(X_train, y_train)

    print('GridSearch CV best score: {:.4f}\n\n'.format(grid_search.best_score_))

    print('Parameters that give the best results: ', '\n\n', (grid_search.best_params_))

    print('\n\nEstimator that was chosen by the search: ', '\n\n', (grid_search.best_estimator_))

