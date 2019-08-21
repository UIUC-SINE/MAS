import pandas as pd

def experiment(func, iterations, **kwargs):
    """
    Run `func` repeatedly and save the results in a Pandas DataFrame
    Args:
        func (function): function which returns a dict of results
           use `return dict(**locals())` to return all locals in function
        iterations (int): number of iterations to repeat experiment
    """

    result = []
    for t in range(iterations):
        print('Trial {}/{}\r'.format(t, iterations), end='')
        result.append(func(**kwargs))

    return pd.DataFrame(result)
