'''
CAPP 30254 1 Machine Learning for Public Policy
Classifier functions for HW3
Spring 2019
Professor: Rayid Ghani
Camilo Arias
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier



def run_model(x_train, y_train, x_test, classif_model, model_params, seed):
    '''
    Create classification model and return y_score. The function takes model,
    which is a function, and a dict with the parameters of that model.
    Inputs:
        x_train: Pandas Series
        y_train: Pandas Series
        x_test: Pandas Series
        classif_model: function
        model_params: dict
    '''

    models_funcitons = {
                        'KNN': build_knn,
                        'decision_tree': build_tree,
                        'logistic_reg': build_logistic_reg,
                        'svm': build_svm,
                        'random_forest': build_random_forest,
                        'gradient_boost': build_gradient_boost,
                        'bagging': build_bagging}
    if classif_model in ['logistic_reg', 'random_forest', 'svm', 'gradient_boost',\
                         'bagging']:
        model_params['seed'] = seed
    build_function = models_funcitons[classif_model]
    model = build_function(y_train, x_train, **model_params)
    y_score = predict_proba(model, x_test)

    return y_score


def build_knn(y_train, x_train, k, weights='uniform', metric='minkowsky'):
    '''
    Build KNN classifier
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        k: integer
        weights: str
        metric: str
    Output
        knn classifier
    '''
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    return knn.fit(x_train, y_train)


def build_tree(y_train, x_train, max_depth, criterion):
    '''
    Fit a decision tree model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        criterion: str
    Output:
        Tree Classifier
    '''

    tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    return tree.fit(x_train, y_train)


def build_logistic_reg(y_train, x_train, C, penalty, fit_intercept, seed):
    '''
    Fit a logistic regression to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        C: float
        penalty: str
        seed: int
    Output:
        lreg Classifier
    '''
    lrf = LogisticRegression(random_state=seed, penalty=penalty, C=C,
                             fit_intercept=fit_intercept, solver='liblinear')
    return lrf.fit(x_train, y_train)


def build_svm(y_train, x_train, C, seed):
    '''
    Fit a SVM to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        C: float
        penalty: str
    Output:
        lreg Classifier
    '''
    svm = LinearSVC(random_state=seed, C=C)
    return svm.fit(x_train, y_train)


def build_random_forest(y_train, x_train, max_depth, criterion, n_estimators,
                        seed):
    '''
    Fit a random forest model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        criterion: str
        n_estimators: int
    Output:
        RF Classifier
    '''
    forest = RandomForestClassifier(max_depth=max_depth, criterion=criterion,
                                    n_estimators=n_estimators, random_state=seed)
    return forest.fit(x_train, y_train)


def build_gradient_boost(y_train, x_train, max_depth, n_estimators,
                        loss, seed):
    '''
    Fit a random forest model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        loss: str
        n_estimators: int
    Output:
        GB Classifier
    '''
    gb = GradientBoostingClassifier(
                max_depth=max_depth, n_estimators=n_estimators,
                loss=loss, random_state=seed)

    return gb.fit(x_train, y_train)


def build_bagging(y_train, x_train, base_estimator, n_estimators, seed):
    '''
    Fit a random forest model to the data provided, given the ycol and the set
    of x cols.
    Inputs:
        y_train: Pandas Series
        x_train: Pandas DataFrame
        max_depth: int
        loss: str
        n_estimators: int
    Output:
        GB Classifier
    '''
    if base_estimator == 'logistic_regression':
    	print('aaa')
    	bg = BaggingClassifier(base_estimator=LogisticRegression(),
    						   n_estimators=n_estimators,crandom_state=seed)
    else:
	    bg = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators,
                               random_state=seed)

    return bg.fit(x_train, y_train)


def predict_proba(classifier, x_test):
    '''
    Predict probability using classifier and text set
    Inputs:
        classifier: SKlearn classifier
        x_test: Pandas DataFrame
    Output:
        Pandas Series
    '''
    if isinstance(classifier, LinearSVC):
    	pred_scores = classifier.decision_function(x_test)
    	return pd.Series(pred_scores)
    pred_scores = classifier.predict_proba(x_test)
    return pd.Series([x[1] for x in pred_scores])


def classify(y_prob, top_k):
    '''
    Classify a Pandas Series given a threshold.
        y_prob: Pandas Series
        threshold: float
    Output:
        Pandas Series
    '''
    threshold = y_prob.quantile(1 - top_k, interpolation='higher')
    y_pred = np.where(y_prob > threshold, 1, 0)
    return y_pred


def build_evaluation_metrics(y_true, y_pred, y_score=None, output_dict=True):
    '''
    Get evaluation metrics of Precision, Recall, F1
    Inputs:
        y_true: Pandas Series
        y_pred: Pandas Series
        y_score: Pandas Series
        Output_dict: bool
    Output:
        dict
    '''
    report_dict = classification_report(y_true, y_pred,
                                        output_dict = output_dict)
    key_metrics = {'precision': report_dict['1']['precision'],
                   'recall':  report_dict['1']['recall'],
                   'f1': report_dict['1']['f1-score'],
                   'accuracy': accuracy_score(y_true, y_pred)}
    if not y_score is None:
        key_metrics['roc auc'] = roc_auc_score(y_true, y_score, average='micro')

    return key_metrics


def build_all_metrics_for_model(y_prob, y_test, top_k):
    '''
    Estimate y_pred and return evaluation metrics for a model.
    Inputs:
        model: Sk learn model
        y_test: PD Series
        x_test: PD Series
        threshold: float
    '''
    y_pred = classify(y_prob, top_k)
    metrics = build_evaluation_metrics(y_test, y_pred, y_prob)
    
    return metrics


def plot_precision_recall(y_true, y_score, x_axis='top_k'):
    '''
    Plot precision and recall curves. If x_axis == 'threshold',
    x axis is decision threshold, if x_axis is 'recall', x_axis
    is recall.

    Inputs:
        y_true: Pandas Series
        y_pred: Pandas Series
        x_axis: (threshold', 'recall') str
    Output:
        map
    '''
    top_ks = np.arange(0, 1.1, 0.1)
    results = [build_evaluation_metrics(y_true, classify(y_score, top_k), y_score)\
               for top_k in top_ks]
    precision = np.array([result['precision'] for result in results])
    recall = np.array([result['recall'] for result in results])
    (recall)
    # threshold = 

    if x_axis == 'top_k':
        plt.step(top_ks, precision, color='b', alpha=0.4,
             where='pre', label='Precision')
        plt.step(top_ks, recall, color='r', alpha=0.4,
             where='pre', label='Recall')
        plt.xlabel('top k')
        plt.ylabel('Value')
        plt.legend()
        plt.ylim([0.0, 1])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve by top k: AP={0:0.2f}'
                    .format(precision.mean()))
    else:
        plt.step(recall, precision, color='b', alpha=0.4 ,
                 where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'
                    .format(precision.mean()))
    plt.show()

