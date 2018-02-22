# kept for backwards compat.
import itertools as itt

from recipes.logging import LoggingMixin
from recipes.dict import AttrDict

from ..modelling import lm_compat as modlib


class ModelData(object):
    def __init__(self, model, n, nfit, folder, clobber=False):
        """Shared data containers for model"""

        folder = Path(folder)
        npars = model.npar

        # fitting parameters
        shape = (n, nfit, npars)
        locPar = folder / 'par'
        locStd = folder / 'parStd'
        self.params = make_shared_mem(locPar, shape, np.nan, clobber)
        # standard deviation on parameters
        self.params_std = make_shared_mem(locStd, shape, np.nan, clobber)

        if hasattr(model, 'integrate'):
            locFlx = folder / 'flx'
            locFlxStd = folder / 'flxStd'
            shape = (n, nfit)
            self.flux = make_shared_mem(locFlx, shape, np.nan, 'f', clobber)
            self.flux_std = make_shared_mem(locFlxStd, shape, np.nan, 'f', clobber)
            # NOTE: can also be computed post-facto



class ModelDb(LoggingMixin):
    """container for model data"""

    @property
    def gaussians(self):
        return [mod for mod in self.models if 'gauss' in str(mod).lower()]

    def __init__(self, model_names):

        self.model_names = model_names
        self.nmodels = len(model_names)
        self.build_models(model_names)
        self._indexer = {}

    def build_models(self, model_names):
        counter = itt.count()
        self.db = AttrDict()

        for name in model_names:
            cls = getattr(modlib, name)
            model = cls()  # initialize
            model.basename = self.basename  # logging!!
            self.db[model.name] = model
            self._indexer[model] = next(counter)

        self.models = list(self._indexer)

    def __getstate__(self):

        # capture what is normally pickled
        state = self.__dict__.copy()
        # since the models from lm_compat are not picklable, we replace their instances with
        # their class names in the following data containers

        for attr in ('_indexer', 'data', 'resData'):
            dic = state.get(attr)
            if dic:
                state[attr] = type(dic)((mod.__class__.__bases__[-1].__name__, val)
                                        for mod, val in dic.items())

        state.pop('db')
        state.pop('models')

        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, state):
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(state)
        # rebuild the models
        # self.build_models(self.model_names)

        # from IPython import embed
        # print('\n' * 10, 'BARF!!', )
        # embed()

        # try:
        # for attr in ('data', 'resData'):
        #     dic = state.get(attr)
        #     for model in self.models:
        #         name = model.__class__.__bases__[-1].__name__
        #         dic[model] = dic.pop(name)
        #         # except Exception as err:
        #         #     print('\n' * 10, 'BARF!!', )
        #         #     embed()

