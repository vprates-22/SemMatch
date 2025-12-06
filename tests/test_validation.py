import pytest

from semmatch.utils.validation import validate_params


def test_validate_params_success():
    params = {'threshold': 2.5, 'method': 'a'}
    rules = {
        'threshold': {'required': True, 'type': (int, float), 'min': 0.0, 'max': 10.0},
        'method': {'required': False, 'type': str, 'choices': ['a', 'b'], 'default': 'a'},
    }

    validated = validate_params(params, rules)
    assert validated['threshold'] == 2.5
    assert validated['method'] == 'a'


def test_validate_params_missing_required():
    params = {'method': 'a'}
    rules = {'threshold': {'required': True, 'type': (int, float)}}
    with pytest.raises(ValueError):
        validate_params(params, rules)


def test_validate_params_type_mismatch():
    params = {'threshold': 'bad'}
    rules = {'threshold': {'required': True, 'type': (int, float)}}
    with pytest.raises(ValueError):
        validate_params(params, rules)


def test_validate_params_range_and_choices():
    params = {'threshold': 20, 'method': 'c'}
    rules = {
        'threshold': {'required': True, 'type': (int, float), 'min': 0.0, 'max': 10.0},
        'method': {'required': True, 'type': str, 'choices': ['a', 'b']},
    }
    with pytest.raises(ValueError):
        validate_params(params, rules)
