from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.compose import make_column_selector as selector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ames_housing = pd.read_csv("../datasets/AmesHousing.csv")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]

#On va s'intéresser uniquement aux variables numériques

numerical_columns_selector = selector(dtype_exclude=object)

numerical_columns = numerical_columns_selector(data)

data_numerical=data[numerical_columns]

#Fin de la récupération des variables numériques

alphas = np.logspace(-3, 3, num=101)

model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas,cv=2)) #Recherche du meilleur paramètre alpha avec deux validations croisées


cv_results = cross_validate(
    model, data_numerical, target, cv=10, return_estimator=True,
    return_train_score=True, scoring="neg_mean_squared_error"
)#Ensuite 10 validations croisées pour évaluer le modèle

#Dessiner un boxplot pour visualiser les poids donnés aux varibles par le modèle RidgeCV

coefs = [est[-1].coef_ for est in cv_results["estimator"]]

weights_ridge = pd.DataFrame(coefs, columns=numerical_columns)

color = {"whiskers": "black", "medians": "black", "caps": "black"}

weights_ridge.plot.box(color=color, vert=False, figsize=(6, 16))

_ = plt.title("Ridge weights")

print("Comme nous le voyns sur le boxplot, les deux variables les plus importantes sont OverallQual and GrLivArea\n ")

#Fin du graphe

scores_train=-cv_results["train_score"]
scores_test=-cv_results["test_score"]

print(f"Erreur quadratique moyenne du modèle de régression linéaire sur les données d'entrainement:\n"
      f"{scores_train.mean():.3f} ± {scores_train.std():.3f}")

print(f"Erreur quadratique moyenne du modèle de régression linéaire sur les données de test:\n"
      f"{scores_test.mean():.3f} ± {scores_test.std():.3f}")

