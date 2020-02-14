# bioactivity_chem_structure_ML
Generating an ensemble classification model from binary features describing chemical structures to evaluate a hypothesis from previous modelling.

## To run the script:

run:
```sh
git clone https://github.com/mshobair/RandomForest_hypothesis.git
cd RandomForest_hypothesis
pipenv install
pipenv run python ./rf_hypothesis.py
```
The output will have balanced accuracy values from cross-validations. Using the ensemble method to probe a null hypothesis, the features seem to perform comparable to the previous model. The previous model is rule-based to maximize the predictive power of integrating locally-bound models.

## The model could be used via a Flask API deployed on Heroku:

run:
```sh
curl -X GET https://flask-rf.herokuapp.com/predict
```
