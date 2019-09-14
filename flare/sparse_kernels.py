import numpy as np
from flare import mc_simple


# -----------------------------------------------------------------------------
#                            two plus three body
# -----------------------------------------------------------------------------

def two_plus_three_env(env1, env2, hyps, cutoffs):
    return mc_simple.two_plus_three_mc_en(env1, env2, hyps, cutoffs)


def two_plus_three_struc(env1, struc1, hyps, cutoffs):
    pass
