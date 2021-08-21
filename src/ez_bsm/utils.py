'''Utilities for dealing with SPX option chains'''

__all__ = []

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

_gaussian = scipy.stats.norm()
_N_cdf = _gaussian.cdf
_N_pdf = _gaussian.pdf

_DAYS_PER_YEAR = 365.

def discount_factor_from(strikes, calls, puts, width=4, divisor=25):
    '''Extract a discount factor from box spreads'''
    pc = pd.DataFrame(dict(Strike=strikes, C=calls, P=puts))
    pc.set_index('Strike', inplace=True)
    pc = pc.loc[[x for x in pc.index if x%divisor==0]].copy()
    pc['Strike'] = pc.index
    pc['Strike2'] = pc.Strike.shift(width)
    pc['C2'] = pc.C.shift(width)
    pc['P2'] = pc.P.shift(width)
    pc['Df'] = ((pc.P - pc.P2) + (pc.C2 - pc.C)) / (pc.Strike - pc.Strike2)
    return float(pc.Df.median())

def underlying_fwd_from_pcp(strikes, calls, puts, discount_factor, width=300):
    '''Extract underlying forward from put call parity'''
    pc = pd.DataFrame(dict(Strike=strikes, C=calls, P=puts))
    pc['F'] = ((pc.C - pc.P) / discount_factor) + pc.Strike
    rng = pc[np.abs(pc.Strike-pc.F.median())<width]
    return float(rng.F.mean())

def greeks_from(fwd, strike, tau, sigma, is_call):
    '''Computes Black-Scholes prices and greeks assuming zero interest and dividend.
    
    Inputs are assumed to be np.array or pandas.Series.

    Returns a dictionary 

    See 
    https://en.wikipedia.org/wiki/Greeks_(finance)
    '''
    sqrt_tau = np.sqrt(tau)
    log_fwd_over_strike = np.log(fwd/strike)
    d1 = (log_fwd_over_strike + 0.5*sigma**2 * tau) / (sigma*sqrt_tau)
    d2 = d1 - sqrt_tau * sigma
    
    Nd1 = _N_cdf(d1)
    phid1 = _N_pdf(d1)
    Nd2 = _N_cdf(d2)
    Nmd1 = _N_cdf(-d1)
    Nmd2 = _N_cdf(-d2)
    
    call_price = fwd * Nd1 - strike * Nd2
    put_price = -fwd * Nmd1 + strike * Nmd2
    
    call_delta = Nd1
    put_delta = -Nmd1
    
    # Note: in this special case, theta is the same for call and put options
    theta = -0.5*fwd * (sigma/sqrt_tau) * phid1 
    gamma = (phid1) / (fwd * sigma * sqrt_tau)
    speed = -(gamma / fwd) * (1. + d1 / (sigma * sqrt_tau))

    vega = fwd * sqrt_tau * phid1
    vanna = (vega/fwd)*(1-d1/(sigma*sqrt_tau))
    vomma = vega * d1 * d2 / sigma
    ultima = (-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)
    
    result = dict(  call_price=call_price,
                    put_price=put_price,
                    call_delta=call_delta,
                    put_delta=put_delta,
                    theta=theta,
                    gamma=gamma,
                    speed=speed,
                    vega=vega,
                    vanna=vanna,
                    vomma=vomma,
                    ultima=ultima,
                )
    
    # todo: fill nan with zero before multiplying by bool
    result['price'] = 1.*is_call * call_price + (1.-is_call)*put_price
    result['delta'] = 1.*is_call * call_delta + (1.-is_call)*put_delta
    result['abs_delta'] = np.abs(result['delta'])

    return result

def refine_iv(tgt, price, vega, vomma, ultima, order=3):

    '''One iteration of Newton-like method for implied vol calculation

    A higher order generalization of Newton's method is supported.
    See https://en.wikipedia.org/wiki/Householder%27s_method

    Params
    ------
    tgt : (np.array) observed option price (calibration target)
    price : (np.array) model-computed price (call or put) given sigma
    vega, vomma, ultima : (np.array) model-computed greeks given sigma
    order : (int) 1=Newton's method, 2-3 are higher order Householder methods

    Returns
    -------
    An array (compatible with tgt) that, when added to the current
    implied volatility, gives an improved estimate. That is,
    iv -> iv + update.

    Notes
    -----
    The paper by Li (2006) provides a useful domain for when this
    type of iteration can be expected to converge:
    |x| <= 0.5, 0 << v <= 1, and |x/v| <= 2,
    where x = log(F/K), F=exp((r-q)*tau)*spot, and v = sqrt(tau)*sigma.

    Generally, starting with a sigma near the upper end of Li's domain
    gives good convergence rates.
    '''
    x = tgt - price
    h = x / vega
    if order==1:
        update = h
    elif order==2:
        update = h * (1./(1 + 0.5*(h)*(vomma/vega)))
    elif order==3:
        update = (h 
                  * (1 + 0.5*(vomma/vega)*h)
                  / (1 + (vomma/vega)*h + (1./6.)*(ultima/vega)*h**2 ))
    else:
        raise ValueError("order must be 1,2 or 3, not {}".format(order))
    return update

def raw_compute_iv(tgt, fwd, strike, tau, is_call,
                   initial_sigma=2., num_iters=12, order=3):
    '''Apply Newton-like iteration to solve for implied vol with no error checks.
    
    '''
    sigma = initial_sigma * (1. + 0*tgt)
    for it in range(num_iters):
        greeks = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma, is_call=is_call)
        update = refine_iv(tgt=tgt, 
                           price=greeks['price'], 
                           vega=greeks['vega'], 
                           vomma=greeks['vomma'], 
                           ultima=greeks['ultima'], 
                           order=order)
        sigma += update
    return sigma

def safe_compute_iv(tgt, fwd, strike, tau, is_call,
                    initial_sigma=1.5,
                    num_iters=12, 
                    order=3,
                    sigma_bounds=(0.01,2.),
                    price_tol=None):
    '''Apply Newton-like iteration to solve for implied vol with some error checking
    
    '''
    greeks_low = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[0], is_call=is_call)
    greeks_high = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[1], is_call=is_call)

    clip_tgt = np.clip(tgt, greeks_low['price'], greeks_high['price'])
    
    iv = raw_compute_iv(clip_tgt, fwd, strike, tau, is_call,
                        initial_sigma=initial_sigma,
                        num_iters=num_iters, order=order)

    greeks = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=iv, is_call=is_call)
    #iv[clip_tgt != tgt] = np.nan # todo: float equality check is sometimes not what we want
    if price_tol is not None:
        iv[np.abs(greeks['price']-tgt)>price_tol] = np.nan
    
    return iv

class SingleExpirationChain:

    def __init__(self, is_call, strike, expiration, is_am, close_date,
                 mark=None, bid=None, ask=None):
        self.frame = pd.DataFrame(dict(
            is_call=np.asarray(is_call), 
            strike=np.asarray(strike),
        ))

        self.is_am = is_am
        self.expiration = pd.Timestamp(expiration)
        self.close_date = pd.Timestamp(close_date)

        self.dte = (self.expiration - self.close_date).days + 1 # todo: is this ok?
        self.tau = self.dte / _DAYS_PER_YEAR

        assert (mark is not None or ((bid is not None) and (ask is not None)))
        if mark is None:
            self.frame['bid'] = np.asarray(bid) 
            self.frame['ask'] = np.asarray(ask) 
            self.frame['mark'] = 0.5*(self.frame.bid + self.frame.ask)
        else:
            self.frame['mark'] = np.asarray(mark)
        
        self.frame['type'] = 'P'
        self.frame.loc[self.frame.is_call,'type'] = 'C'

        # these are expensive, so don't compute them unless they are needed
        self._discount_factor = None
        self._int_rate = None
        self._implied_forward = None
        self._greeks = None

        self.pc_frame = (self.frame.pivot('strike','is_call')
                             .rename(axis=1, level=1, mapper=(lambda x: 'C' if x else 'P')))
    
    def compute_all(self):
        # this should cause everything else to be computed, so fast access through private members is ok
        _ = self.greeks

    def as_dataframe(self):
        df = self.greeks.copy()
        df['is_am'] = self.is_am
        df['dte'] = self.dte
        df['tau'] = self.tau
        df['expiration'] = self.expiration
        df['close_date'] = self.close_date
        df['discount_factor'] = self.discount_factor
        df['int_rate'] = self.int_rate
        df['implied_forward'] = self.implied_forward
        return df

    @property
    def discount_factor(self):
        if self._discount_factor is not None:
            return self._discount_factor
        self._discount_factor = discount_factor_from(self.pc_frame.index, 
                                                    self.pc_frame.mark.C, 
                                                    self.pc_frame.mark.P)
        return self._discount_factor
        
    @property
    def int_rate(self):
        if self._int_rate is not None:
            return self._int_rate
        self._int_rate = float(-1*np.log(self.discount_factor) / self.tau)
        return self._int_rate

    @property
    def implied_forward(self):
        if self._implied_forward is not None:
            return self._implied_forward
        self._implied_forward = underlying_fwd_from_pcp(strikes=self.pc_frame.index, 
                                                        calls=self.pc_frame.mark.C, 
                                                        puts=self.pc_frame.mark.P, 
                                                        discount_factor=self.discount_factor)
        return self._implied_forward

    @property
    def greeks(self):
        if self._greeks is not None:
            return self._greeks
        self._greeks = self.frame.copy()
        self._greeks['implied_vol'] = safe_compute_iv(
                                    self.frame.mark, 
                                    self.implied_forward, 
                                    self.frame.strike, 
                                    self.tau, 
                                    self.frame.is_call)
        self._all_greeks = greeks_from(fwd=self.implied_forward,
                                strike=self.frame.strike, 
                                tau=self.tau, 
                                sigma=self._greeks.implied_vol,
                                is_call=self.frame.is_call)
        for c in ['abs_delta','delta','gamma','vega','theta']:
            self._greeks[c] = self._all_greeks[c]
        return self._greeks
                                



