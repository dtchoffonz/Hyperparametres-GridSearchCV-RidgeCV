from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import pandas as pd

penguins = pd.read_csv("../datasets/penguins.csv")

#Nous utiliserons ces trois variables
columns = ["Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"]
target_name = "Species"

# Remove lines with missing values for the columns of interest
penguins_non_missing = penguins[columns + [target_name]].dropna()

data = penguins_non_missing[columns]
target = penguins_non_missing[target_name]

#Très important KNeighborsClassifier se base sur le calcul des distances,
#donc un prétraitement des variables rend ce modèle plus efficace
model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=101)),
])

#Liste des hyparamètres
all_preprocessors = [
    None,
    StandardScaler(),
    MinMaxScaler(),
    QuantileTransformer(n_quantiles=100),
    PowerTransformer(method="box-cox"),
]
  #Quels prétraitement est-il le plus adapté???

param_grid ={"preprocessor": all_preprocessors,
             "classifier__n_neighbors": [5, 51, 101]}
#Fin de la partie des hyparamètres

model_grid_search = GridSearchCV(model, param_grid=param_grid,n_jobs=2, cv=2)#deux(02) validations croisées internes pour déterminer les meilleurs hyparamètres

scores = cross_validate(
    model_grid_search, data, target, cv=10, n_jobs=2, return_estimator=True
)#dix(10) validations croisées externes pour déterminer les performences du modèles

cv_test_scores = scores['test_score']

#Liste des meilleurs hyparamètres pour chaque validation croisée
for cv_fold, estimator_in_fold in enumerate(scores["estimator"]):
    print(
        f"Meilleurs hyperparamètres pour la validation croisée #{cv_fold + 1}:\n"
        f"{estimator_in_fold.best_params_}"
    )
