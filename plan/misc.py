def sortmore(*args, **kw):
    """
    Extends builtin list sorting with ability to to sorts any number of lists 
    simultaneously according to:
        * optional sorting key function(s) 
        * and/or a global sorting key function.

    Parameters
    ----------
    One or more lists

    Keywords
    --------
    globalkey: None
        revert to sorting by key function
    globalkey: callable
        Sort by evaluated value for all items in the lists
        (call signature of this function needs to be such that it accepts an
        argument tuple of items from each list.
        eg.: globalkey = lambda *l: sum(l) will order all the lists by the
        sum of the items from each list

    if key: None
        sorting done by value of first input list
        (in this case the objects in the first iterable need the comparison
        methods __lt__ etc...)
    if key: callable
        sorting done by value of key(item) for items in first iterable
    if key: tuple
        sorting done by value of (key[0](item_0), ..., key[n](item_n)) for items in
        the first n iterables (where n is the length of the key tuple)
        i.e. the first callable is the primary sorting criterion, and the
        rest act as tie-breakers.

    Returns
    -------
    Sorted lists
    
    Raises
    ------
    ValueError, KeyError
    
    Examples
    --------
    Capture sorting indeces:
        l = list('CharacterS')
        In [1]: sortmore( l, range(len(l)) )
        Out[1]: (['C', 'S', 'a', 'a', 'c', 'e', 'h', 'r', 'r', 't'],
                 [0, 9, 2, 4, 5, 7, 1, 3, 8, 6])
        In [2]: sortmore( l, range(len(l)), key=str.lower )
        Out[2]: (['a', 'a', 'C', 'c', 'e', 'h', 'r', 'r', 'S', 't'],
                 [2, 4, 0, 5, 7, 1, 3, 8, 9, 6])
    """
    #TODO: extend examples doc
    
    farg = list(args[0])
    if not len(farg):
        return args
    
    globalkey   =       kw.get('globalkey')
    key         =       kw.get('key')
    order       =       kw.get('order')
    
    #enable default behaviour
    if key is None:
        if globalkey:
            key = lambda x: 0               #if global sort function given and no local (secondary) key given, ==> no tiebreakers
        else:
            key = lambda x: x               #if no global sort and no local sort keys given, sort by item values
    if globalkey is None:
        globalkey = lambda *x: 0
    
    #validity checks for sorting functions
    if not isinstance(globalkey, coll.Callable):
        raise ValueError( 'globalkey needs to be callable' )
        
    if isinstance(key, coll.Callable):
        _key = lambda x: (globalkey(*x), key(x[0]))
    elif isinstance(key, tuple):
        key = (k if k else lambda x: 0 for k in key)
        _key = lambda x : (globalkey(*x),) + tuple(f(z) for (f,z) in zip(key, x))
    else:
        raise KeyError(("Keyword arg 'key' should be 'None', callable, or a" 
                        "sequence of callables, not {}").format(type(key)) )
    
    res = sorted(list(zip(*args)), key=_key)
    if order:
        if order == -1 or order.startswith(('descend', 'reverse')):
            res = reversed(res)
    
    return tuple(map(list, zip(*res)))

#====================================================================================================
def sorter(*args, **kw):
    '''alias for sortmore'''
    return sortmore(*args, **kw)