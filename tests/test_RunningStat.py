import numpy as np 

from ppo_pytorch.utils import RunningStat


def test_update():
    stat = RunningStat()
    x = np.random.uniform(size=10)
    stat.update(x)
    y = x
    np.testing.assert_almost_equal(stat.mean, y.mean())
    np.testing.assert_almost_equal(stat.std, y.std())

    x2 = np.random.uniform(size=5)
    stat.update(x2)
    y = np.concatenate([x, x2])
    np.testing.assert_almost_equal(stat.mean, y.mean())
    np.testing.assert_almost_equal(stat.std, y.std())

    x3 = np.random.uniform(size=100)
    stat.update(x3)
    y = np.concatenate([x, x2, x3])
    np.testing.assert_almost_equal(stat.mean, y.mean())
    np.testing.assert_almost_equal(stat.std, y.std())
