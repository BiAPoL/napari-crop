def array_allclose_in_list(array, list_arrays):
    '''Check if array is in list of arrays'''
    import numpy as np
    return next((True for elem in list_arrays if elem.size == array.size and np.allclose(elem, array, atol=0.05)), False)


def find_array_allclose_position_in_list(array, list_arrays):
    '''Find position of array in list of arrays'''
    import numpy as np
    return next((i for i, elem in enumerate(list_arrays) if elem.size ==
                array.size and np.allclose(elem, array, atol=0.05)), None)
