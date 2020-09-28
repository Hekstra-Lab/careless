from careless.models.priors.base import Prior
from careless.utils.distributions import Stacy
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import numpy as np


def Centric(**kw):
    """
    According to Herr Rupp, the pdf for centric normalized structure factor 
    amplitudes, E, is:
    P(E) = (2/pi)**0.5 * exp(-0.5*E**2)

    In common parlance, this is known as halfnormal distribution with scale=1.0

    RETURNS
    -------
    dist : tfp.distributions.HalfNormal
        Centric normalized structure factor distribution
    """
    dist = tfd.HalfNormal(scale=1., **kw)
    return dist

def Acentric(**kw):
    """
    According to Herr Rupp, the pdf for acentric normalized structure factor 
    amplitudes, E, is:
    P(E) = 2*E*exp(-E**2)

    This is exactly a Raleigh distribution with sigma**2 = 1. This is also
    the same as a Chi distribution with k=2 which has been transformed by 
    rescaling the argument. 

    RETURNS
    -------
    dist : tfp.distributions.TransformedDistribution
        Centric normalized structure factor distribution
    """
    dist = tfd.Weibull(2., 1., **kw)
    return dist

class WilsonPrior(Stacy, Prior):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, centric, epsilon):
        """
        Parameters
        ----------
        centric : array
            Floating point array with value 1. for centric reflections and 0. for acentric.
        epsilon : array
            Floating point array with multiplicity values for each structure factor.
        """
        #Stacy parameters
        centric_params  = (np.sqrt(2.).astype(np.float32), 0.5, 2.)
        acentric_params = (1., 1., 2.)

        self.epsilon = np.array(epsilon, dtype=np.float32)
        self.centric = np.array(centric, dtype=np.float32)

        theta = centric_params[0] * self.centric + acentric_params[0] * (1. - self.centric)
        alpha = centric_params[1] * self.centric + acentric_params[1] * (1. - self.centric)
        beta  = centric_params[2] * self.centric + acentric_params[2] * (1. - self.centric)

        theta = theta * np.sqrt(self.epsilon) # I am pretty sure this is right.

        super().__init__(theta, alpha, beta)


