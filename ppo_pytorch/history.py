import numpy as np




class TrainHistory(object):

    def __init__(self):
        self.policy_loss = []
        self.value_loss = []

    def __len__(self):
        len(self.policy_loss)

    def collect_entry_stats(self, losses):
        stats = {'epochs': len(losses) if losses else 0}
        stats['min'] = np.min(losses) if losses else 0
        stats['max'] = np.max(losses) if losses else 0
        stats['mean'] = np.mean(losses) if losses else 0
        stats['std'] = np.std(losses) if losses else 0
        stats['last'] = losses[-1] if losses else 0
        return stats

    def update(self, attr, losses):
        entry = self.collect_entry_stats(losses)
        attr.append(entry)

    def update_policy_losses(self, losses):
        self.update(self.policy_loss, losses)

    def update_value_losses(self, losses):
        self.update(self.value_loss, losses)

    @property
    def last(self):
        return self.policy_loss[-1], self.value_loss[-1]

    def sub_format(self, d):
        formatter = '{}: {:.3f}'
        return ' '.join([formatter.format(k, v) for k, v in d.items() if k != 'epochs'])

    def __str__(self):
        formatter = '{} loss : {}'
        p_str = formatter.format(
            'policy', self.sub_format(self.policy_loss[-1]))
        v_str = formatter.format('value', self.sub_format(self.value_loss[-1]))
        return '{} | {}'.format(p_str, v_str)

