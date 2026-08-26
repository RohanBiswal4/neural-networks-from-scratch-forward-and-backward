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

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.
    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.
    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)
    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    D={}
    def forward(x):
      cache=x
      y=np.maximum(x,0)
      return y,cache
    def backward(dout,cache):
      mask=np.where(cache>0,1.0,0)
      dx=dout*mask
      return dx,{}
    D['params']={}
    D['forward']=forward
    D['backward']=backward
    return D

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    if scheme=='he':
      sigma=np.sqrt(2/in_dim)
    elif scheme == 'Xavier' or scheme=='Glorot':
      sigma==1/np.sqrt(in_dim)
    b=np.zeros((out_dim))
    W=np.random.normal(loc=0,scale=sigma,size=(in_dim,out_dim))
    return W,b

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    def softmax(x):
      m=np.max(x,axis=1,keepdims=True)
      x1=x-m
      x1=np.exp(x1)
      s=np.sum(x1,axis=1,keepdims=True)
      return x1/s 
    def loss_fn(logits,labels):
      B=len(labels)
      X=softmax(logits)
      grad=np.zeros_like(logits)
      loss=np.mean(-np.log(X[np.arange(len(labels)), labels]))
      grad[np.arange(B),labels]=(X[np.arange(B), labels]-1)/B
      grad=np.where(grad==0,X/B,grad)
      return loss,grad
    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    def forward(x):
      C=[]
      for l in layers:
        x,c=l['forward'](x)
        C.append(c)
      y=x
      return y,C[::-1]
    def backward(dout,caches):
      grads_list=[]
      for l,c in zip(layers[::-1],caches):
        (dx, grads_dict)= l['backward'](dout,c)
        dout=dx
        grads_list.append(grads_dict)
      return dout,grads_list[::-1]
    return {
        'params': [layer['params'] for layer in layers],
        'forward': forward,
        'backward': backward
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.
    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels
    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    param_grads=[]
    (logits, caches)=model['forward'](x)
    (loss, d_logits)=loss_fn(logits, y)
    (dx, param_grads)=model['backward'](d_logits,caches)
    return loss,param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    def step(grads):
      for i,j in zip(params,grads):
        if isinstance(i,dict):
          if i=={}:
            continue
          i["W"]-=lr*j["W"]
          i['b']-=lr*j['b']
        elif isinstance(i,list):
          if i==[]:
            continue
          i[0]-=lr*j[0]
          i[1]-=lr*j[1]
        else:
          i-=lr*j
    return {'step':step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.
    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    logits,caches=model['forward'](x_batch)
    loss,dlogits=loss_fn(logits,y_batch)
    dx,param_grads=model['backward'](dlogits, caches)
    optimizer['step'](param_grads)
    return loss

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

