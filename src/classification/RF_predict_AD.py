import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

# input data
X = pd.read_csv(r'./data/X_expr.csv').drop(['Unnamed: 0', 'seqLibID'], axis=1)
y = pd.read_csv(r'./data/y_cog.csv').drop(['Unnamed: 0', 'seqLibID'], axis=1)

# remove outliers identified from PCA only
outliers = [18, 132, 172]
X = X.drop(outliers, axis=0)  # drop outliers identified from PCA
y = y.drop(outliers, axis=0)  # drop outliers identified from PCA

# drop genes/samples with all zero values
X[X < 0] = 0.0
X = X.loc[(X.sum(axis=1) != 0), (X.sum(axis=0) != 0)]  # select sum !=0 rows and columns
y = y.loc[X.index]

# prepare data
X = X.to_numpy()
y = y.values.flatten()
y_copy = y.copy()
run = 0
fold = 5
AUC, ACC = np.zeros(fold*3), np.zeros(fold*3)

for diseases in ['AD', 'MildCognitiveImpairment', 'NoCognitiveImpairment']:
    # use one vs. all others strategy
    if diseases == 'AD':
        y[y_copy == 'MildCognitiveImpairment'] = 0
        y[y_copy == 'NoCognitiveImpairment'] = 0
        y[y_copy == 'AD'] = 1
    if diseases == 'MildCognitiveImpairment':
        y[y_copy == 'AD'] = 0
        y[y_copy == 'NoCognitiveImpairment'] = 0
        y[y_copy == 'MildCognitiveImpairment'] = 1
    else:
        y[y_copy == 'MildCognitiveImpairment'] = 0
        y[y_copy == 'AD'] = 0
        y[y_copy == 'NoCognitiveImpairment'] = 1

    # transform y values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.ravel())

    # stratify data for cross-validation
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kf.split(X, y_copy)):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        # standardize data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # calculate p-values for each genes
        pval = pd.DataFrame(np.ones(X_train.shape[1]), index=np.arange(0, X_train.shape[1], 1),
                            columns=['p_value'])
        # select relevant features
        for k in np.arange(0, X_train.shape[1], 1):
            one = X_train[y_train == 1, k]
            others = X_train[y_train == 0, k]
            _, p = stats.f_oneway(one, others)
            pval.loc[k, 'p_value'] = p
        pval = pval.fillna(1)
        Sig = pval.loc[(pval['p_value'] < 0.1)]

        # Feature importance based on feature permutation
        forest = RandomForestClassifier(random_state=0, class_weight='balanced')
        forest.fit(X_train, y_train)
        forest_importances = pd.Series(forest.feature_importances_)

        # select overlap genes (p<0.1 and importance by RF)
        overlap = list(set(Sig.index.to_list()) & set(forest_importances.index[forest_importances > 0].tolist()))
        X_train = X_train[:, overlap]
        X_test = X_test[:, overlap]

        # tuning RF prediction parameters by exhaust grid search to optimize a prediction RF model
        start_time = time.time()
        param_grid = {'n_estimators': np.arange(50, 200, 10),
                    'max_features': np.arange(0.2, 1, 0.1),
                    'max_depth': np.arange(1, 10, 1),
                    'max_samples': np.arange(0.2, 1, 0.1)}
        model = GridSearchCV(RandomForestClassifier(criterion='gini', random_state=0, class_weight='balanced'),
                                            param_grid, n_jobs=-1).fit(X_train, y_train)
        model = model.best_estimator_
        elapsed_time = time.time() - start_time
        print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

        # predict using the model built
        test_prob = model.predict_proba(X_test)
        test_predict = model.predict(X_test)
        AUC[run] = roc_auc_score(y_test, test_prob[:, 1])
        ACC[run] = accuracy_score(y_test, test_predict)
        print("Test accuracy: %f" % ACC[run])
        print("Test      AUC: %f" % AUC[run])

        # prepare the result output
        if run == 0:
            test_out = pd.DataFrame(test_prob)
            test_out['predict'] = test_predict
            test_out['class_label'] = y_test
        else:
            temp = pd.DataFrame(test_prob)
            temp['predict'] = test_predict
            temp['class_label'] = y_test
            pd.concat([test_out, temp], axis=0)
        run = run + 1

# write results to output files
test_out.to_csv("All_prediction_probability_final.csv")
performance_out = pd.DataFrame(np.arange(0, fold*3, 1))
performance_out['AUC'] = AUC
performance_out['ACC'] = ACC
performance_out.to_csv("Performance_result_final.csv")
