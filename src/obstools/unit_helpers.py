# std
import inspect

# third-party
from loguru import logger
from astropy import units as u

# local
from recipes.decorators import Decorator


# ---------------------------------------------------------------------------- #
NULL = object()
POS, PKW, VAR, KWO, VKW = inspect._ParameterKind

# ---------------------------------------------------------------------------- #


def get_value(x):
    """Get numeric value from quantity or return object itself."""
    return getattr(x, 'value', x)


def has_unit(val):
    """Check if object has a physically meaningful unit."""
    return not no_unit(val)


def no_unit(val):
    return getattr(val, 'unit', None) in (None, u.dimensionless_unscaled)


def get_unit_string(val, default='', style='latex'):
    if hasattr(val, 'unit'):
        unit = val.unit
    elif isinstance(val, u.UnitBase):
        unit = val
    else:
        return default

    return unit.to_string(style).strip('$')


def _repr_type(kls):
    return f'{kls.__module__}.{kls.__name__}'


class default_units(Decorator):
    """
    Decorator for applying default units to function input parameters.
    """

    def __init__(self, default_units=(), **kws):
        self.sig = None     # placeholder
        self.variadic_default = None
        self.default_units = dict(default_units, **kws)
        for name, unit in self.default_units.items():
            if not isinstance(unit, u.UnitBase):
                raise TypeError(
                    f'Default unit for parameter {name!r} should be of type '
                    f'{_repr_type(u.UnitBase)!r} not '
                    f'{_repr_type(type(unit))!r}.'
                )

    def __call__(self, func):
        self.sig = inspect.signature(func)
        return super().__call__(func)

    def __wrapper__(self, func, *args, **kws):
        ba = self.sig.bind(*args, **kws).arguments
        if vkw := next(
            (p for p in self.sig.parameters.values() if p.kind is VKW), {}
        ):
            # func has variadic kws
            self.variadic_default = self.default_units.pop(vkw.name, None)
            vkw = ba.pop(vkw.name, {})

        return func(
            *(() if (_self := ba.pop('self', NULL)) is NULL else (_self,)),
            **self.apply(**ba, **vkw)
        )
    

    def _apply_args(self, args):
        for i, (val, unit) in enumerate(zip(args, self.default_units.values())):
            if no_unit(val) and unit:
                logger.info('Applying default unit {} to positional argument {}'
                            ' in function {}', unit, i, self.__wrapped__)
                yield val * unit
            yield val

    def _apply_kws(self, kws):
        for name, val in kws.items():
            unit = self.default_units.get(name, self.variadic_default)
            if has_unit(val) or unit is None:
                yield name, val
            else:
                yield name, val * unit  # NOTE: >> does not copy underlying data

    def apply(self, *args, **kws):
        # positional params
        if args:
            return (next, tuple)[len(args) > 1](self._apply_args(args))

        # keyword params
        return dict(self._apply_kws(kws))


def _check_optional_units(namespace, allowed_physical_types):
    for name, kinds in allowed_physical_types.items():
        if name not in namespace:
            continue

        obj = namespace[name]
        if isinstance(kinds, (str, u.PhysicalType)):
            kinds = (kinds, )

        if isinstance(obj, u.Quantity) and (obj.unit.physical_type not in kinds):
            raise u.UnitTypeError(f'Parameter {name} should have physical '
                                  f'type(s) {kinds}, not '
                                  f'{obj.unit.physical_type}.')
        # else:
        #     logger.info
