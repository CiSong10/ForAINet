"""To help debug"""

import pickle


def save_pickle(variables, file_path, scope=None):
    """
    Save a list of variables (by name) to a pickle file.
    
    Args:
        variables (list of str): List of variable names as strings.
        file_path (str): Path to the pickle file.
        scope (dict, optional): The namespace to pull variables from. 
                                Defaults to globals().
    """
    if scope is None:
        scope = locals()
    
    state = {}
    for var in variables:
        if var in scope:
            state[var] = scope[var]
        else:
            raise ValueError(f"Variable '{var}' not found in provided scope.")
    
    with open(file_path, "wb") as f:
        pickle.dump(state, f)


def load_pickle(file_path, scope=None):
    """
    Load variables from a pickle file and inject them into the given scope.
    
    Args:
        file_path (str): Path to the pickle file.
        scope (dict, optional): The namespace to load variables into. 
                                Defaults to globals().
    """
    if scope is None:
        scope = locals()
    
    with open(file_path, "rb") as f:
        state = pickle.load(f)

    for key, value in state.items():
        scope[key] = value
