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
    for dirname, _, filenames in os.walk('C:\\Users\\Justin\ML Final\\archive'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    warnings.filterwarnings('ignore')

    data = 'C:\\Users\\Justin\\ML Final\\archive\\ParisHousingClass99.88.csv'
    df = pd.read_csv(data, header=0, sep=',')

    # print(df.head)

    X = df.drop(['category'], axis=1)
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_train.isnull().sum())
    # print(X_test.isnull().sum())

    cols = X_train.columns
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_test = scaler.transform(X=X_test)

    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    # print(X_train.head())

    #
    # Check Accuracy Score
    #
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    # print(y_pred)
    # print('Model Accuracy Score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    #
    # Compare the train-set and test-set accuracy
    #
    y_pred_train = gnb.predict(X_train)
    # print(y_pred_train)
    # print('Training-set Accuracy Score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))

    #
    # Check for Over-fitting and Under-fitting
    #
    # print('Training set Score: {:.4f}'.format(gnb.score(X_train, y_train)))
    # print('Test set Score: {:.4f}'.format(gnb.score(X_test, y_test)))

    #
    # Compare model accuracy with null accuracy
    #
    # print(y_test.value_counts())

    null_accuracy = (2615 / (2615 + 385))
    # print('Null Accuracy Score: {0:0.4f}'.format(null_accuracy))

    #
    # Confusion Matrix
    #
    cm = confusion_matrix(y_test, y_pred)

    # print('Confusion Matrix:\n\n', cm)
    # print('\nTrue Positives(TP) = ', cm[0, 0])
    # print('\nTrue Negatives(TN) = ', cm[1, 1])
    # print('\nFalse Positives(FP) = ', cm[0, 1])
    # print('\nFalse Negatives(FN) = ', cm[1, 0])

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    # sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlOrBr')
    # plt.show()

    #
    # Classification Metrics
    #
    # print(classification_report(y_test, y_pred))
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    # print('Classification Accuracy: {0:0.4f}'.format(classification_accuracy))

    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    # print('Classification Error: {0:0.4}'.format(classification_error))

    #
    # Precision
    #
    precision = TP / float(TP + FP)
    # print('Precision: {0:0.4}'.format(precision))

    #
    # Recall
    #
    recall = TP / float(TP + FN)
    # print('Recall or Sensetivity: {0:0.4}'.format(recall))

    #
    # True Positive Rate
    #
    true_positive_rate = TP / float(TP + FN)
    # print('True Positive Rate: {0:0.4}'.format(true_positive_rate))

    #
    # False Positive Rate
    #
    false_positive_rate = FP / float(FP + TN)
    # print('False Positive Rate: {0:0.4}'.format(false_positive_rate))

    #
    # Specificity
    #
    specificity = TN / float(TN + FP)
    # print('Specificity: {0:0.4}'.format(specificity))

    #
    # Probabilities
    #
    y_pred_prob = gnb.predict_proba(X_test)[0:10]
    # print(y_pred_prob)

    y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of basic', 'Prob of luxury'])
    # print(y_pred_prob_df)

    # print(gnb.predict_proba(X_test)[0:10, 1])
    y_pred1 = gnb.predict_proba(X_test)[:, 1]

    # plt.rcParams['font.size'] = 12
    # plt.hist(y_pred1, bins=10)
    # plt.title('Histogram of Predicted Probabilities of Luxury')
    # plt.xlim(0,1)
    # plt.xlabel('Predicted Probabilities of Luxury')
    # plt.ylabel('Frequency')
    # plt.show()

    #
    # ROC - AUC
    #
    fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label='luxury')

    # plt.figure(figsize=(6, 4))
    # plt.plot(fpr, tpr, linewidth=2)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.rcParams['font.size'] = 12
    # plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Quality')
    # plt.xlabel('False Positive Rate (1 - Specificity)')
    # plt.ylabel('True Positive Rate (Sensitivity)')
    # plt.show()

    ROC_AUC = roc_auc_score(y_test, y_pred1)
    #print('ROC AUC: {:.4}'.format(ROC_AUC))

    Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()
    # print('Cross Validated ROC_AUC: {:.4f}'.format(Cross_validated_ROC_AUC))

    #
    # K-Fold Cross Validation
    #
    scores = cross_val_score(gnb, X_train, y_train, cv=10, scoring='accuracy')
    # print('Cross Validation Scores: {}'.format(scores))
    # print('Average Cross-Validation Score: {:.4}'.format(scores.mean()))






