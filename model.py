"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad=np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        f_p=x.copy()
        f_m=x.copy()
        f_p[idx]+=eps
        f_m[idx]-=eps
        grad[idx]= (f(f_p)-f(f_m))/(2*eps)
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    a=np.abs(analytic_grad)
    b=np.abs(numeric_grad)
    rel= np.max(np.abs(analytic_grad-numeric_grad)/np.maximum(np.maximum(a,b),tol))
    return rel

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.
    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W,b=weight_init_fn(in_dim,out_dim)
    D={}
    D['params']={}
    D['params']["W"],D['params']["b"]=W,b
    def forward(x):
      y=x@ D['params']["W"] + D['params']["b"]
      cache=x
      return y,cache
    def backward(dout,cache):
      dx=dout@ D['params']["W"].T
      dw=cache.T@ dout
      db=np.sum(dout,axis=0)
      grads={'W':dw,
              'b':db}
      return (dx,grads)
    D['backward']=backward
    D['forward']=forward
    return D

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

