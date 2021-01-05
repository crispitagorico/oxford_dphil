import numpy as np
import tjl_hall_numpy_lie as li


def _get_keys(dimension, degree):
    """Get keys as tuples"""

    s = li.sigkeys(dimension, degree)
    tuples = []
    for t in s.split():
        if len(t) > 2:
            t = t.replace(")", ",)")

        tuples.append(eval(t))

    return tuples


def merge_dicts(d1, d2):
    """Merges two dictionaries into one."""

    # Create a copy for safety
    d = d1.copy()

    for key, value in d2.items():
        d[key] = d.get(key, 0.) + value

    return d

def concatenate(u, list_words):
    """Concatenates a letter with each word in a list."""

    return {tuple([u] + list(word)): count for word, count in list_words.items()}

def basis_shuffle(w1, w2):
    """Computes the shuffle product of two words."""

    if len(w1) == 0:
        return {w2: 1.}

    if len(w2) == 0:
        return {w1: 1.}



    word_dict1 = concatenate(w1[0], basis_shuffle(w1[1:], w2))
    word_dict2 = concatenate(w2[0], basis_shuffle(w1, w2[1:]))
    
    return merge_dicts(word_dict1, word_dict2)
    
def dict_shuffle(word_dict1, word_dict2, depth=np.inf):
    """Computes the shuffle product of formal sums of words.

    Parameters
    ----------
    word_dict1 : dictionary
        Dictionary whose keys are words, which is given by
        a tuple, and whose values are scalars.
    word_dict2 : dictionary
        Dictionary whose keys are words, which is given by
        a tuple, and whose values are scalars.
    depth : int or np.inf, optional
        Order of the truncated tensor algebra. If it is
        equal to np.inf, the full tensor algebra is
        considered.
        Default is np.inf.
    
    Returns
    -------
    dictionary
        Shuffle product, in the format of {word: scalar}.

    Examples
    --------
    >>> dict_words1 = {(1, 2): 1.}
    >>> dict_words2 = {(2, 3): 1., (1, 1): 1.}
    >>> dict_shuffle(dict_words1, dict_words2)
    {(1, 1, 2, 1): 2.0, (2, 1, 3, 2): 1.0, (2, 3, 1, 2): 1.0,
    (2, 1, 2, 3): 1.0, (1, 1, 1, 2): 3.0, (1, 2, 3, 2): 1.0,
    (1, 2, 1, 1): 1.0, (1, 2, 2, 3): 2.0}

    """
    
    product = {}

    for word1, value1 in word_dict1.items():
        for word2, value2 in word_dict2.items():
            if len(word1) + len(word2) > depth:
                continue

            word1_shuffle_word2 = basis_shuffle(word1, word2)
            for word, value in word1_shuffle_word2.items():
                word1_shuffle_word2[word] = value * value1 * value2

            product = merge_dicts(product, word1_shuffle_word2)

    return product


def shuffle_product(tensor1, tensor2, dimension, degree):
    """Computes the shuffle product of two tensors.

    We identify the truncated tensor algebra with its dual
    using an inner product on the truncated tensor algebra,
    so that it makes sense to consider the shuffle product
    of tensor1 and tensor2.

    """

    sigkeys = _get_keys(dimension, degree)
    
    word_dict1 = {}
    word_dict2 = {}

    # Transform tensors to dictionaries
    for key, scalar1, scalar2 in zip(sigkeys, tensor1, tensor2):
        if scalar1 != 0.:
            word_dict1[key] = scalar1
        
        if scalar2 != 0.:
            word_dict2[key] = scalar2

    
    tensor1_shuffle_tensor2 = dict_shuffle(word_dict1, word_dict2, depth=degree)

    # We now convert the dict to an array

    product = np.zeros(len(sigkeys))

    for i, key in enumerate(sigkeys):
        product[i] = tensor1_shuffle_tensor2.get(key, 0.)


    return product
    



    
    