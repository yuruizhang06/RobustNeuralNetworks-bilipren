# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Unitary layer using Cayley transform.
This layer applies a learned orthogonal (unitary) transformation to the input
using the Cayley map, preserving 2-norms in the transformation process.

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (May 2024)
'''
import jax.numpy as jnp
from flax import linen as nn 
from flax.struct import dataclass
from robustnn.utils import cayley
from flax.typing import Array, PrecisionLike, Dtype

@dataclass
class DirectOrthogonalParams:
    """Data class to keep track of implicit params for Orthogonal layer."""
    W: Array
    a: Array
    b: Array

@dataclass
class ExplicitOrthogonalParams:
    """Data class to keep track of explicit params for Orthogonal layer."""
    R: Array
    b: Array


class Unitary(nn.Module):
    """Unitary linear transformation layer using a Cayley transform.

    This layer applies a learned orthogonal (unitary) transformation to the input
    using the Cayley map, preserving 2-norms in the transformation process.

    Example usage::

        >>> layer = Unitary(input_size=4)
        >>> x = jnp.ones((1, 4))
        >>> params = layer.init(jax.random.key(0), x)
        >>> y = layer.apply(params, x)

    Attributes:
        input_size: Size of the input features.
        use_bias: Whether to include a learnable bias term (default: True).
    """

    input_size: int
    use_bias: bool = True 

    def setup(self):
        """Setup method for the Unitary layer."""
        
        m = self.input_size 

        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (m, self.input_size),
                       jnp.float32)
        a = self.param('a', 
                       nn.initializers.constant(jnp.linalg.norm(W)), 
                       (1,),
                       jnp.float32)
        
        if self.use_bias: 
            b = self.param('b', nn.initializers.zeros_init(), (m,), jnp.float32)
        else:
            b = 0.

        self.direct = DirectOrthogonalParams(W=W, a=a, b=b)

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        '''
        Call method for the Unitary layer.
        This method applies the Cayley transform to the input tensor `x` and
        returns the transformed tensor `z`.
        The transformation is defined as:
            z = x @ R^T
        where `R` is the Cayley matrix obtained from the learned parameters.
        The transformation is designed to be orthogonal, preserving the 2-norm of the input.
        The Cayley matrix is computed using the learned weight matrix `W` and a scaling factor `a`.
        The weight matrix `W` is initialized using the Glorot normal initializer,
        and the scaling factor `a` is initialized to the norm of `W`.
        If `use_bias` is set to True, a learnable bias term is added to the output.
        The parameters `W`, `a`, and `b` are learned during training.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            z: Output tensor of shape (batch_size, output_dim).
        '''
        explict = self._direct_to_explicit()
        return self._explicit_call(x, explict)
    
    def _direct_to_explicit(self) -> ExplicitOrthogonalParams:
        """Convert implicit parameters to explicit parameters."""
        W = self.direct.W
        a = self.direct.a
        R = cayley((a / jnp.linalg.norm(W)) * W)
        b = self.direct.b 
        return ExplicitOrthogonalParams(R=R, b=b)
    
    def _explicit_call(self, x: jnp.array, e: ExplicitOrthogonalParams) -> Array:
        """Call method for the Unitary layer using explicit parameters.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            e: ExplicitOrthogonalParams object containing explicit parameters.
        Returns:
            z: Output tensor of shape (batch_size, output_dim).
        """
        R = e.R
        b = e.b
        z = x @ R.T 
        if self.use_bias: 
            z += b
        return z
    
    def _explicit_inverse_call(self, y: jnp.array, e: ExplicitOrthogonalParams) -> Array:
        """
        Inverse call method for the Unitary layer using explicit parameters.
        Args:
            y: Output tensor of shape (batch_size, output_dim).
            e: ExplicitOrthogonalParams object containing explicit parameters.
        Returns:
            x: Input tensor of shape (batch_size, input_dim).
        """
        R = e.R
        b = e.b 
        if self.use_bias: 
            y -= b
        
        x = y @ R
        return x
    
    
    #################### Convenient Wrappers ####################
    def inverse_call(self, params: dict, y: Array, explicit: ExplicitOrthogonalParams):
        """Evaluate the inverse of the explicit model for an orthogonal layer.

        Args:
            params (dict): Flax model parameters dictionary.
            y (Array): model outputs.
            explicit (ExplicitOrthogonalParams): explicit params (inverse it the same as forward).

        Returns:
            Array: model inputs.
        """
        return self.apply(params, y, explicit, method="_explicit_inverse_call")

    def explicit_call(self, params: dict, x: Array, explicit: ExplicitOrthogonalParams):
        """Evaluate the explicit model for an orthogonal layer.

        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): model inputs.
            explicit (ExplicitLBDNParams): explicit params.

        Returns:
            Array: model outputs.
        """
        return self.apply(params, x, explicit, method="_explicit_call")
    
    def direct_to_explicit(self, params: dict)-> ExplicitOrthogonalParams:
        """Convert from direct orthogonal layer params to explicit form for eval.

        Args:
            params (dict): Flax model parameters dictionary.
            
        Returns:
            ExplicitOrthogonalParams: explicit orthogonal layer params.
        """
        return self.apply(params, method="_direct_to_explicit")
    
    # def get_params(self)-> ExplicitOrthogonalParams:
    #     """Get explicit parameters for the Unitary layer."""
    #     self.explict = self._direct_to_explicit()
    #     R = self.explict.R
    #     b = self.explict.b

    #     params = {
    #         'R': R,
    #         'b': b
    #     }
    #     return params


@dataclass
class DirectDynOrthogonalParams:
    """Data class to keep track of direct params for a dynamic orthogonal layer."""
    X: Array
    bx: Array
    by: Array


@dataclass
class ExplicitDynOrthogonalParams:
    """Data class to keep track of explicit params for a dynamic orthogonal layer."""
    A: Array
    B: Array
    C: Array
    D: Array
    bx: Array
    by: Array


class DynUnitary(nn.Module):
    """Dynamic (stateful) orthogonal layer using a Cayley transform.

    Unlike `Unitary`, this layer is a small *recurrent* linear system whose
    stacked state-space matrix is orthogonal (hence norm-preserving, so it does
    not change the bi-Lipschitz bounds of a network):

        [x1 - bx;  y - by] = G @ [x;  u],     G = [[A, B], [C, D]]  (orthogonal)

    i.e. in row-vector form

        x1 = x @ A.T + u @ B.T + bx     (next state)
        y  = x @ C.T + u @ D.T + by     (output)

    Because `G` is orthogonal, the system is exactly invertible *given the next
    state* `x1` (a non-causal inverse): `[x; u] = G.T @ [x1 - bx;  y - by]`.

    Attributes:
        input_size: input/output feature size (nu == ny).
        state_size: dimension of the internal state.
        use_bias: whether to include the learnable biases bx, by (default: True).
        param_dtype: dtype for the parameters (default: float32).
    """
    input_size: int
    state_size: int
    use_bias: bool = True
    param_dtype: Dtype = jnp.float32

    def setup(self):
        nx, ny = self.state_size, self.input_size
        k = nx + ny
        X = self.param('X', nn.initializers.glorot_normal(), (k, k), self.param_dtype)
        if self.use_bias:
            bx = self.param('bx', nn.initializers.zeros_init(), (nx,), self.param_dtype)
            by = self.param('by', nn.initializers.zeros_init(), (ny,), self.param_dtype)
        else:
            bx = jnp.zeros((nx,), self.param_dtype)
            by = jnp.zeros((ny,), self.param_dtype)
        self.direct = DirectDynOrthogonalParams(X=X, bx=bx, by=by)

    def __call__(self, state: Array, u: Array):
        """Forward pass: (state, inputs) -> (next_state, outputs)."""
        e = self._direct_to_explicit()
        return self._explicit_call(state, u, e)

    def _direct_to_explicit(self) -> ExplicitDynOrthogonalParams:
        nx = self.state_size
        G = cayley(self.direct.X)
        A = G[:nx, :nx]
        B = G[:nx, nx:]
        C = G[nx:, :nx]
        D = G[nx:, nx:]
        return ExplicitDynOrthogonalParams(A, B, C, D, self.direct.bx, self.direct.by)

    def _explicit_call(self, x: Array, u: Array, e: ExplicitDynOrthogonalParams):
        x1 = x @ e.A.T + u @ e.B.T + e.bx
        y = x @ e.C.T + u @ e.D.T + e.by
        return x1, y

    def _explicit_inverse_call(self, x_next: Array, y: Array,
                               e: ExplicitDynOrthogonalParams):
        """Non-causal inverse: recover (prev_state, inputs) from
        (next_state, outputs) using the orthogonality of G."""
        xb = x_next - e.bx
        yb = y - e.by
        x_prev = xb @ e.A + yb @ e.C
        u = xb @ e.B + yb @ e.D
        return x_prev, u

    @nn.nowrap
    def initialize_carry(self, rng, input_shape):
        """Initialise the dynamic-orthogonal state (zeros)."""
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.state_size,)
        return jnp.zeros(mem_shape, self.param_dtype)

    #################### Convenient Wrappers ####################

    def direct_to_explicit(self, params: dict) -> ExplicitDynOrthogonalParams:
        """Convert direct params to explicit params."""
        return self.apply(params, method="_direct_to_explicit")

    def explicit_call(self, params: dict, x: Array, u: Array,
                      explicit: ExplicitDynOrthogonalParams):
        """Evaluate the forward layer given explicit params."""
        return self.apply(params, x, u, explicit, method="_explicit_call")

    def inverse_call(self, params: dict, x_next: Array, y: Array,
                     explicit: ExplicitDynOrthogonalParams):
        """Evaluate the non-causal inverse given explicit params."""
        return self.apply(params, x_next, y, explicit, method="_explicit_inverse_call")