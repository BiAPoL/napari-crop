def array_allclose_in_list(array, list_arrays):
    '''Check if array is in list of arrays'''
    import numpy as np
    return next((True for elem in list_arrays if elem.size == array.size and np.allclose(elem, array, atol=0.05)), False)


def find_array_allclose_position_in_list(array, list_arrays):
    '''Find position of array in list of arrays'''
    import numpy as np
    return next((i for i, elem in enumerate(list_arrays) if elem.size ==
                array.size and np.allclose(elem, array, atol=0.05)), None)


def compute_combined_slices(array_shape, slices1, slices2):
    import numpy as np

    combined_slices = []

    for dim, (slice1, slice2) in enumerate(zip(slices1, slices2)):
        # Adjust slice indices based on the shape of the array
        adjusted_slice1 = slice1.indices(array_shape[dim])
        adjusted_slice2 = slice2.indices(array_shape[dim])

        # Compute the combined slice
        start = adjusted_slice1[0] + adjusted_slice2[0]
        stop = min(adjusted_slice1[1], adjusted_slice1[0] + adjusted_slice2[1])
        step = adjusted_slice1[2] * adjusted_slice2[2]
        combined_slice = slice(start, stop, step)

        combined_slices.append(combined_slice)

    return combined_slices
