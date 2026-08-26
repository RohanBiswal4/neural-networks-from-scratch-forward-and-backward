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
    sigma=1
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

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.
    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    rng=np.random.default_rng(seed)
    history=[]
    n=len(y)
    k=n//batch_size
    idx=np.arange(n)
    rng.shuffle(idx)
    x=x[idx]
    y=y[idx]
    for t in range(epochs):
      loss=0.00
      for i in range(0,n,batch_size):
        x_batch=x[i:i+batch_size,:]
        y_batch=y[i:i+batch_size]
        loss+=train_step(model, loss_fn, optimizer, x_batch, y_batch)
      history.append(loss/k)
    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.
    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    rng=np.random.default_rng(seed)
    def make_xor_dataset(n, input_dim, rng):
      # Generate exactly input_dim features
      X = rng.normal(0, 0.2, size=(n, input_dim))
      # Randomly choose one of four XOR quadrants
      quadrant = rng.integers(0, 4, size=n)
      signs = np.array([
          [-1, -1],
          [ 1,  1],
          [-1,  1],
          [ 1, -1]
      ])
      # Apply XOR structure only to the first two features
      X[:, :2] += signs[quadrant]
      # Label based on which XOR region the point belongs to
      y = (X[:, 0] * X[:, 1] < 0).astype(int)
      return X, y
    X,y=make_xor_dataset(1000, input_dim,rng)
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]
    x_train=X[:800]
    y_train=y[:800]
    x_eval=X[800:]
    y_eval=y[800:]
    def he_init(in_dim, out_dim):
      return initialize_weights(in_dim, out_dim, scheme='he')
    weight_init_fn=he_init
    layer1=make_dense(input_dim, 32, weight_init_fn)
    layer2=make_activation(kind='relu')
    layer3=make_dense(32, 64, weight_init_fn)
    layer4=make_activation(kind='relu')
    layer5=make_dense(64, num_classes, weight_init_fn)
    layers=[layer1,layer2,layer3,layer4,layer5]
    loss_fn=make_loss(kind='cross_entropy')
    model=make_sequential(layers)
    optimizer=make_optimizer(model['params'], lr=1e-2, kind='sgd')
    history=train(model, loss_fn, optimizer, x_train, y_train, 1000, 64, seed)
    logits,_=model['forward'](x_eval)
    pred=np.argmax(logits,axis=1)
    acc=np.sum(pred==y_eval)/len(y_eval)
    metrics={'accuracy':acc,
              'x':x_eval,
              'y':y_eval}
    return model,metrics

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

