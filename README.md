# bioactivity_chem_structure_ML
Generating an ensemble classification model from binary features describing chemical structures to evaluate a hypothesis from previous modelling.

To run the script:

clone the repository locally
run: 
pipenv install
python ./RF_hypothesis.py

The output will have balanced accuracy values from cross-validations. Using the ensemble method to probe a null hypothesis, the features seem to perform comparable to the previous model. The previous model is rule-based to maximize the predictive power of integrating locally-bound models.

