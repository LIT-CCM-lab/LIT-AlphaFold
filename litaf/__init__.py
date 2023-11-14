'''
Package containing the code to run LIT-AF, the Laboratoire d'Innovation Therapeutique custom version of AlphaFold.

* *creat_input*, function to create input objects for AF
* *objects*, objects describing monomers and multimers
* *predict_structure*, functions to run the calculations
* *query_pdb*, functions to filter template information
* *utils*, utility functions
* *datatype*, specfic datatypes of the module
* *pipeline*, modified AF data pipeline functions
'''

import litaf.create_input
import litaf.objects
import litaf.predict_structure
import litaf.querypdb
import litaf.utils
import litaf.datatypes
import litaf.pipeline
