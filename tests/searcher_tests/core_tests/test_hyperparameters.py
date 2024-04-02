# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import pytest

from autorecsys.searcher.core import hyperparameters as hp_module


def test_base_hyperparameter():
    base_param = hp_module.HyperParameter(name='base', default=0)
    assert base_param.name == 'base'
    assert base_param.default == 0
    assert base_param.get_config() == {'name': 'base', 'default': 0}
    base_param = hp_module.HyperParameter.from_config(
        base_param.get_config())
    assert base_param.name == 'base'
    assert base_param.default == 0


def test_hyperparameters():
    hp = hp_module.HyperParameters()
    assert hp.values == {}
    assert hp.space == []
    hp.Choice('choice', [1, 2, 3], default=2)
    assert hp.values == {'choice': 2}
    assert len(hp.space) == 1
    assert hp.space[0].name == 'choice'
    hp.values['choice'] = 3
    assert hp.get('choice') == 3
    hp = hp.copy()
    assert hp.values == {'choice': 3}
    assert len(hp.space) == 1
    assert hp.space[0].name == 'choice'
    with pytest.raises(ValueError, match='Unknown parameter'):
        hp.get('wrong')


def test_name_collision():
    # TODO: figure out how name collision checks
    # should work.
    pass


def test_name_scope():
    hp = hp_module.HyperParameters()
    hp.Choice('choice', [1, 2, 3], default=2)
    with hp.name_scope('scope1'):
        hp.Choice('choice', [4, 5, 6], default=5)
        with hp.name_scope('scope2'):
            hp.Choice('choice', [7, 8, 9], default=8)
        hp.Int('range', min_value=0, max_value=10, default=0)

    assert hp.values == {
        'choice': 2,
        'scope1/choice': 5,
        'scope1/scope2/choice': 8,
        'scope1/range': 0
    }
    assert hp.get_value_in_nested_format() == {
        'choice': 2,
        'scope1': {'choice': 5,
                   'scope2': {'choice': 8},
                   'range': 0,
                   },
        }


def test_parent_name():
    hp = hp_module.HyperParameters()
    hp.Choice('a', [1, 2, 3], default=2)
    b1 = hp.Int(
        'b', 0, 10, parent_name='a', parent_values=1, default=5)
    b2 = hp.Int(
        'b', 0, 100, parent_name='a', parent_values=2, default=4)
    assert b1 is None
    assert b2 == 4
    assert hp.values == {
        'a': 2,
        'a=1/b': 5,
        'a=2/b': 4
    }


def test_conditional_scope():
    hp = hp_module.HyperParameters()
    hp.Choice('choice', [1, 2, 3], default=2)
    with hp.conditional_scope('choice', [1, 3]):
        child1 = hp.Choice('child_choice', [4, 5, 6])
    with hp.conditional_scope('choice', 2):
        child2 = hp.Choice('child_choice', [7, 8, 9])
    assert hp.values == {
        'choice': 2,
        'choice=1,3/child_choice': 4,
        'choice=2/child_choice': 7
    }
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert child1 is None
    # Assignment to an active conditional hyperparameter returns the value.
    assert child2 == 7


def test_nested_conditional_scopes_and_name_scopes():
    hp = hp_module.HyperParameters()
    a = hp.Choice('a', [1, 2, 3], default=2)
    with hp.conditional_scope('a', [1, 3]):
        b = hp.Choice('b', [4, 5, 6])
        with hp.conditional_scope('b', 6):
            c = hp.Choice('c', [7, 8, 9])
            with hp.name_scope('d'):
                e = hp.Choice('e', [10, 11, 12])
    with hp.conditional_scope('a', 2):
        f = hp.Choice('f', [13, 14, 15])

    assert hp.values == {
        'a': 2,
        'a=1,3/b': 4,
        'a=1,3/b=6/c': 7,
        'a=1,3/b=6/d/e': 10,
        'a=2/f': 13
    }
    # Assignment to an active conditional hyperparameter returns the value.
    assert a == 2
    assert f == 13
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert b is None
    assert c is None
    assert e is None


def test_get_with_conditional_scopes():
    hp = hp_module.HyperParameters()
    hp.Choice('a', [1, 2, 3], default=2)
    assert hp.get('a') == 2
    with hp.conditional_scope('a', 2):
        assert hp.get('a') == 2


def test_Choice():
    choice = hp_module.Choice('choice', [1, 2, 3], default=2)
    choice = hp_module.Choice.from_config(choice.get_config())
    assert choice.default == 2
    assert choice.random_sample() in [1, 2, 3]
    assert choice.random_sample(123) == choice.random_sample(123)
    # No default
    choice = hp_module.Choice('choice', [1, 2, 3])
    assert choice.default == 1
    with pytest.raises(ValueError, match='default value should be'):
        hp_module.Choice('choice', [1, 2, 3], default=4)


@pytest.mark.parametrize(
    "values,ordered_arg,ordered_val",
    [([1, 2, 3], True, True),
     ([1, 2, 3], False, False),
     ([1, 2, 3], None, True),
     (['a', 'b', 'c'], False, False),
     (['a', 'b', 'c'], None, False)])
def test_Choice_ordered(values, ordered_arg, ordered_val):
    choice = hp_module.Choice('choice', values, ordered=ordered_arg)
    assert choice.ordered == ordered_val
    choice_new = hp_module.Choice(**choice.get_config())
    assert choice_new.ordered == ordered_val


def test_Choice_ordered_invalid():
    with pytest.raises(ValueError, match='must be `False`'):
        hp_module.Choice('a', ['a', 'b'], ordered=True)


def test_Choice_types():
    values1 = ['a', 'b', 0]
    with pytest.raises(TypeError, match='can contain only one'):
        hp_module.Choice('a', values1)
    values2 = [{'a': 1}, {'a': 2}]
    with pytest.raises(TypeError, match='can contain only `int`'):
        hp_module.Choice('a', values2)


def test_Float():
    # Test with step arg
    linear = hp_module.Float(
        'linear', min_value=0.5, max_value=9.5, default=9.)
    linear = hp_module.Float.from_config(linear.get_config())
    assert linear.default == 9.
    assert 0.5 <= linear.random_sample() < 9.5
    assert isinstance(linear.random_sample(), float)
    assert linear.random_sample(123) == linear.random_sample(123)

    # No default
    linear = hp_module.Float(
        'linear', min_value=0.5, max_value=9.5)
    assert linear.default == 0.5


def test_sampling_arg():
    f = hp_module.Float('f', 1e-20, 1e10, sampling='loguniform')
    f = hp_module.Float.from_config(f.get_config())
    assert f.sampling == 'loguniform'

    i = hp_module.Int('i', 0, 10, sampling='uniform')
    i = hp_module.Int.from_config(i.get_config())
    assert i.sampling == 'uniform'

    with pytest.raises(ValueError, match='`sampling` must be one of'):
        hp_module.Int('j', 0, 10, sampling='invalid')


def test_sampling_random_state():
    f = hp_module.Float('f', 1e-3, 1e3, sampling='loguniform')
    rand_sample = f.random_sample()
    assert rand_sample >= f.min_value
    assert rand_sample <= f.max_value

    def log_scale(x, min_value, max_value):
        return math.log(x/min_value) / math.log(max_value/min_value)

    x = 1e-1
    min_value, max_value = 1e-10, 1e10
    # Scale x to [0, 1].
    x_scaled = log_scale(x, min_value, max_value)
    # Scale back.
    x_rescaled = hp_module._log_sample(x_scaled, min_value, max_value)
    assert np.allclose(x, x_rescaled)

    f = hp_module.Float('f', 1e-3, 1e3, sampling='uniform')
    rand_sample = f.random_sample()
    assert rand_sample >= f.min_value
    assert rand_sample <= f.max_value


def test_Int():
    rg = hp_module.Int(
        'rg', min_value=5, max_value=9, default=6)
    rg = hp_module.Int.from_config(rg.get_config())
    assert rg.default == 6
    assert 5 <= rg.random_sample() < 9
    assert isinstance(rg.random_sample(), int)
    assert rg.random_sample(123) == rg.random_sample(123)
    # No default
    rg = hp_module.Int(
        'rg', min_value=5, max_value=9)
    assert rg.default == 5


def test_Boolean():
    # Test default default
    boolean = hp_module.Boolean('bool')
    assert boolean.default is False
    # Test default setting
    boolean = hp_module.Boolean('bool', default=True)
    assert boolean.default is True
    # Wrong default type
    with pytest.raises(ValueError, match='must be a Python boolean'):
        hp_module.Boolean('bool', default=None)
    # Test serialization
    boolean = hp_module.Boolean('bool', default=True)
    boolean = hp_module.Boolean.from_config(boolean.get_config())
    assert boolean.default is True
    assert boolean.name == 'bool'

    # Test random_sample
    assert boolean.random_sample() in {True, False}
    assert boolean.random_sample(123) == boolean.random_sample(123)


def test_merge():
    hp = hp_module.HyperParameters()
    hp.Int('a', 0, 100)
    hp.Float('b', min_value=0.5, max_value=9.5, default=2)

    hp2 = hp_module.HyperParameters()
    hp2.Int('a', 3, 4, default=3)
    hp.Int('c', 10, 100, default=30)
    hp.merge(hp2)

    assert hp.get('a') == 3
    assert hp.get('b') == 2
    assert hp.get('c') == 30

    hp3 = hp_module.HyperParameters()
    hp3.Float('a', 3.5, 4.5)
    hp3.Choice('d', [1, 2, 3], default=1)

    hp.merge(hp3, overwrite=False)

    assert hp.get('a') == 3
    assert hp.get('b') == 2
    assert hp.get('c') == 30
    assert hp.get('d') == 1


def _sort_space(hps):
    space = hps.get_config()['space']
    return sorted(space, key=lambda hp: hp['config']['name'])
