import numpy as np

class LinearInterpolant:   
    def __init__(self, position, N_points):
        self.truncated_index = int(np.floor(position*(N_points-1))) + 1 # skip fixed endpoint
        self.upper_fraction = position*(N_points-1) - self.truncated_index
        self.lower_fraction = 1 - self.upper_fraction

class ExcitationParameters:
    def __init__(self, position, height, width):
        self.position = position
        self.height = height
        self.width = width

class LossyStiffString:
    def __init__(self, f0, Fs, f_loss, T_60, excitation_parameters, is_plucked, inharmonicity):
        self.f0 = f0
        self.Fs = Fs
        self.f_loss = f_loss
        self.T_60 = T_60
        self.excitation_parameters = excitation_parameters
        self.is_plucked = is_plucked
        self.inharmonicity = inharmonicity
        self.N = None
        self.stability = None
        self.u2 = None
        self.u1 = None
        self.M2 = None
        self.M1 = None

        self.build_scheme()
    
    def build_scheme(self): # 1D wave equation with stiffness: ü = cˆ2*u'' - kappaˆ2*u''''
        c = 2*self.f0
        kappa = c*np.sqrt(self.inharmonicity)/np.pi

        z = (np.sqrt(c**4 + (2*kappa*2*np.pi*np.array(self.f_loss))**2) - c**2)/(2*kappa**2)
        coeff = 6*np.log(10)/(z[1]-z[0])
        s_0, s_1 = coeff*(z[1]/self.T_60[0] - z[0]/self.T_60[1]), coeff*(1/self.T_60[1] - 1/self.T_60[0])

        k = 1/self.Fs
        h = np.sqrt( 0.5*((c*k)**2 + 4*s_1*k + np.sqrt( ( (c*k)**2 + 4*s_1*k )**2 + ( 4*kappa*k )**2 ) ) )
        self.N = int(np.floor(1/h)) # try to divide string length evenly for grid
        h = 1/self.N # re-assign

        lambd, mu = c*k/h, kappa*k/h**2
        self.stability = lambd**2 + 4*mu**2

        # initialization
        u2 = np.zeros((self.N+1))
        if self.is_plucked:
            u2[1:-1] = self.triangle( self.N-1, self.excitation_parameters ) # initialise u2 (state 2 time steps ago) according to clamped B.C.
            u1 = u2.copy() # initialise u1 (state 1 time step ago) using zero initial velocity
        else: # strike
            v0 = np.zeros((self.N+1))
            v0[1:-1] = self.raised_sine( self.N-1, self.excitation_parameters ) # zero initial displacement (u2), specify initial velocity v0
            u1 = k*v0 # initialise u1 (state 1 time step ago) using initial velocity v0
        self.u2 = u2
        self.u1 = u1

        # define scheme coefficients
        c1 = 2*(1-lambd**2-3*mu**2) - 4*s_1*k/h**2
        c2 = lambd**2 + 4*mu**2 + 2*s_1*k/h**2
        c3 = -mu**2
        c4 = s_0*k + 4*s_1*k/h**2 - 1
        c5 = -2*s_1*k/h**2
        d = 1/(s_0*k + 1) # RHS divisor

        # specify beginning/end of free spatial gridpoints (exclude 1, 2, N & N+1)
        l_i, l_f = 2, self.N-1
        C1 = c1 * d * np.ones((l_f - l_i + 1))
        C2 = c2 * d * np.ones((l_f - l_i))
        C3 = c3 * d * np.ones((l_f - l_i - 1))
        C4 = c4 * d * np.ones((l_f - l_i + 1))
        C5 = c5 * d * np.ones((l_f - l_i))

        # assemble update matrices
        self.M1 = np.diag(C3,-2)+np.diag(C2,-1)+np.diag(C1,0)+np.diag(C2,1)+np.diag(C3,2)
        self.M2 = np.diag(C5,-1)+np.diag(C4,0)+np.diag(C5,1) # sparse()

    def triangle( self, N, pluck ):
        '''Produces a pluck displacement vector - 'pos' (0,1), 'h' in m'''
        ind = np.round( pluck.position*N )
        x = np.hstack(( np.arange( 0, 1, 1/(ind-1) ), np.arange( 1, -1/N, -1/(N-ind) ) )) # start, stop 1-1/(ind-1), step
        return pluck.height*x

    def raised_sine( self, N, pluck ):
        '''raised sinusoidal function to initialize string with'''
        x_ctr = np.pi * (0.5 - pluck.position/pluck.width); 
        y = 0.5 * pluck.height * ( np.sin( np.pi * np.arange(N) / (pluck.width*(N-1)) + x_ctr ) + 1 )
        y[ : int( np.round( N*(pluck.position-pluck.width) ) ) + 1 ] = 0
        y[ int( np.round( N*(pluck.position+pluck.width) ) ) + 1 : ] = 0
        return y

def interpolated_read(x, linear_interpolant):
    return linear_interpolant.lower_fraction*x[linear_interpolant.truncated_index] + linear_interpolant.upper_fraction*x[linear_interpolant.truncated_index+1]