
class MissingDependency(Exception):
    pass


def plot_rewards(agent):
    try:
        import pandas as pd
    except ImportError as e:
        raise MissingDependency("{}. The plot_rewards method requires pandas (because dev was lazy) if you want you can 'pip install pandas'".format(e))
    df = pd.DataFrame()
    df['reward'] = pd.Series(agent.episode_rewards)
    df['mean_100_episide_reward'] = df.reward.rolling(window=100, min_periods=0).mean()
    return df.plot()


