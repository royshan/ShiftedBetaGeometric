import pymc as pm
from pymc.distributions import normal_like as gauss
import numpy as np
import pandas

# actual fresh data!
data = pandas.read_csv('../data/new_data.csv')

total_size = 500
index = np.arange(data.shape[0])
np.random.shuffle(index)
index = index[:total_size]

data = data.iloc[index]
data.index = np.arange(total_size)

data['age'] = data['age'].values.astype(int)
data['alive'] = data['alive'].values.astype(int)

x = data[['creative', 'trades', 'monthly']].values
names = ['bias'] + list(data.keys()[:-2])

alpha_bias = pm.Normal('alpha_bias', -1, 1, observed=False, size=1)
beta_bias = pm.Normal('beta_bias', 1, 1, observed=False, size=1)

alpha = pm.Normal('alpha', 0, .25, observed=False, size=x.shape[1])
beta = pm.Normal('beta', 0, .25, observed=False, size=x.shape[1])

global_bool_mask = np.zeros((x.shape[0], 2), dtype=bool)
global_bool_mask[:, 1][data.alive.values==1] = True
global_bool_mask[:, 0][data.alive.values==0] = True


@pm.deterministic
def _recursive_retention_stats(alpha=alpha,
                               beta=beta,
                               alpha_bias=alpha_bias,
                               beta_bias=beta_bias,
                               num_periods=data.age.values,
                               features=x):
    """
    A function to calculate the expected probabilities recursively.
    Using equation 7 from [1] and the alpha and beta coefficients
    obtained by training this model, it computes P(T = t) recursively,
    returning a list with all values.
    Survival function recursive calculation. Using equation 7 from [1]
    and the alpha and beta coefficients obtained by training this
    model, it computes S(T = t) recursively, returning a list of all
    computed values.. To do so it must first invoke the function
    P_T_is_t calculate the monthly churn rates for the given time
    window, and then use it to compute the survival curve recursively.
    :param alpha: float
        The distribution for the alpha parameter.
    :param beta: float
        The distribution for the beta parameter.
    :param num_periods: Int
        The number of periods for which the probability of churning
        should be computed.
    :return: (list, list)
        A list with probability of churning for all periods from month
        zero to num_periods.
    """
    stats = np.zeros((num_periods.shape[0], 2))

    for entry, vals in enumerate(zip(num_periods, features)):

        period, predictors = vals

        # get coeffs from combinaitons
        a = np.exp(sum(alpha_bias + predictors * alpha))
        b = np.exp(sum(beta_bias + predictors * beta))

        p_old = None
        p_new = a / (a + b)

        s_old = 1
        s_new = s_old - p_new

        for t in range(2, period + 1):

            # New values for old values
            p_old = p_new
            s_old = s_new

            # Compute latest p value and append
            p_new = (b + t - 2.) / (a + b + t - 1.) * p_old

            # use the most recent appended p value to keep building s
            s_new = s_old - p_new

        stats[entry][0] = p_new
        stats[entry][1] = s_old

    return stats


@pm.observed
def retention_rates(stats=_recursive_retention_stats,
                    value=data.alive.values):

    def logp(stats, value):
        return np.log(stats[global_bool_mask]).sum()


model = pm.Model([alpha, beta, alpha_bias, beta_bias,
                  _recursive_retention_stats, retention_rates])
mcmc = pm.MCMC(model)


mcmc.sample(2e2, 1e1, thin=1)

df_alpha = pandas.DataFrame()
df_alpha['bias'] = mcmc.trace('alpha_bias')[:][:, 0]

df_beta = pandas.DataFrame()
df_beta['bias'] = mcmc.trace('beta_bias')[:][:, 0]

for i in range(x.shape[1]):
    df_alpha['alpha_%i' % i] = mcmc.trace('alpha')[:][:, i]
    df_beta['beta_%i' % i] = mcmc.trace('beta')[:][:, i]

df_alpha.to_csv('alpha_traces.csv', index=False)
df_beta.to_csv('beta_traces.csv', index=False)