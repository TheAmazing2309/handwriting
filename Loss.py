import tensorflow as tf

epsilon = 1e-6

def loss(pi, mux, muy, sigmax, sigmay, rho, penup, target, mask) -> float:
    """
    Negative log-likelihood loss for the Mixture Density Network 
    """
    pi     = pi[:, :-1, :]
    mux    = mux[:, :-1, :]
    muy    = muy[:, :-1, :]
    sigmax = sigmax[:, :-1, :]
    sigmay = sigmay[:, :-1, :]
    rho    = rho[:, :-1, :]
    penup  = penup[:, :-1, :]
    target = target[:, 1:, :]
    mask   = mask[:, 1:]

    dx = tf.expand_dims(target[:,:,0], -1)
    dy = tf.expand_dims(target[:,:,1], -1)
    accpenUp = target[:,:,2]
    Z = ((((dx-mux)/sigmax)**2)+(((dy-muy)/sigmay)**2))-(2*rho*(dx-mux)*(dy-muy)/(sigmax*sigmay))
    log_N = (-Z / (2*(1-rho**2+epsilon))
             - tf.math.log(2*3.14159265)
             - tf.math.log(sigmax)
             - tf.math.log(sigmay)
             - 0.5*tf.math.log(1-rho**2+epsilon))
    P = tf.reduce_logsumexp(tf.math.log(pi+epsilon) + log_N, axis=-1)
    penup = tf.squeeze(penup, axis=-1)
    penloss = -(accpenUp*tf.math.log(penup+epsilon)+(1-accpenUp)*tf.math.log(1-penup+epsilon))

    mask = tf.cast(mask, tf.float32)
    perStepLoss = mask * (-P + penloss)
    perSequenceLoss = tf.reduce_sum(perStepLoss, axis=1)
    return tf.reduce_mean(perSequenceLoss)