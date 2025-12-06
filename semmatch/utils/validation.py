"""
Module: utils.validation
-----------------------

Utilities for lightweight parameter validation used across the SemMatch project.

The functions here provide a small, dependency-free alternative to full
schema-validation libraries. They are intended for validating simple
configuration dictionaries or function parameters according to a compact set
of rules (required, type, range, choices).

Example
-------
rules = {
    'threshold': {'required': True, 'type': (int, float), 'min': 0.0, 'max': 10.0},
    'method': {'required': False, 'type': str, 'choices': ['a','b'], 'default': 'a'},
}

validated = validate_params(params, rules)
"""
from abc import ABCMeta
from typing import Any, Dict, Union

from semmatch.configs.base import Config


class ValidatedClass(metaclass=ABCMeta):
    """
    Base class for classes that validate their configuration parameters.

    Subclasses should define a class-level `_validation_rules` dictionary
    specifying the validation rules for their configuration parameters.

    During initialization, the provided configuration dictionary is validated
    against these rules using the `validate_params` function.
    """

    _validation_rules: Dict[str, Dict] = {}

    def __init__(self, config: Union[Config, Dict[str, Any]] = None):
        # Construir regras efetivas mesclando _validation_rules de toda a MRO
        effective_rules: Dict[str, Dict] = {}
        for cls in reversed(type(self).__mro__):
            rules = getattr(cls, '_validation_rules', None)
            if not rules:
                continue
            # mescla rasa: chaves novas ou sobrepõe regras de classes mais base
            for k, v in rules.items():
                effective_rules[k] = {**effective_rules.get(k, {}), **v}

        self.config = validate_params(config or {}, effective_rules)


def _type_name(t):
    # Formata nomes de tipos para mensagens de erro.
    # Aceita um único tipo, tuple ou lista de tipos.
    if isinstance(t, (list, tuple)):
        names = []
        for x in t:
            try:
                names.append(x.__name__)
            except Exception:
                names.append(str(x))
        return ', '.join(names)
    try:
        return t.__name__
    except Exception:
        return str(t)


def validate_params(params: Dict[str, Any], rules: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate a dict of parameters against a compact rules specification.

    Parameters
    ----------
    params : dict
        Input parameters to validate.
    rules : dict
        Mapping from parameter name to validation rule dict. Each rule may
        contain the following keys:
        - `required` (bool): whether the parameter must be present (default False)
        - `type` (type or tuple[type,...]): expected Python type(s)
        - `min` (numeric): minimum allowed value (inclusive)
        - `max` (numeric): maximum allowed value (inclusive)
        - `choices` (iterable): allowed set of values
        - `default`: default value to use when parameter is missing and not required
        - `allow_none` (bool): whether None is accepted as a valid value

    Returns
    -------
    dict
        A new dict with validated (and defaulted) parameters.

    Raises
    ------
    ValueError
        If any parameter fails validation. The exception message contains
        details on all detected validation issues.
    """
    params = params or {}
    errors = []
    out: Dict[str, Any] = {}

    for key, rule in rules.items():
        required = rule.get('required', False)
        default = rule.get('default', None)
        allow_none = rule.get('allow_none', False)

        if key not in params:
            if required and default is None:
                errors.append(f"Missing required parameter '{key}'")
                continue
            else:
                out[key] = default
                continue

        val = params[key]
        if val is None:
            if allow_none:
                out[key] = None
                continue
            else:
                errors.append(
                    f"Parameter '{key}' is None but None is not allowed")
                continue

        if 'type' in rule:
            expected = rule['type']
            # Permitir que o usuário forneça uma lista de tipos além de um type/tuple
            if isinstance(expected, list):
                expected = tuple(expected)
            if not isinstance(val, expected):
                errors.append(
                    f"Parameter '{key}' expected type {_type_name(expected)} but got {type(val).__name__}")
                continue

        if 'choices' in rule:
            choices = rule['choices']
            if val not in choices:
                errors.append(
                    f"Parameter '{key}' has invalid value {val!r}; allowed: {list(choices)}")
                continue

        if ('min' in rule) or ('max' in rule):
            try:
                vnum = float(val)
            except Exception:
                errors.append(
                    f"Parameter '{key}' could not be interpreted as a number for range check")
                continue
            if 'min' in rule and vnum < rule['min']:
                errors.append(
                    f"Parameter '{key}' value {vnum} < min {rule['min']}")
                continue
            if 'max' in rule and vnum > rule['max']:
                errors.append(
                    f"Parameter '{key}' value {vnum} > max {rule['max']}")
                continue

        out[key] = val

    if errors:
        raise ValueError("Parameter validation failed:\n  " +
                         "\n  ".join(errors))

    return out
