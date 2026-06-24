import tensorflow as tf

epsilon = 1e-6

def loss(pi, mux, muy, sigmax, sigmay, rho, penup, target):
    dx = tf.expand_dims(target[:,:,0], -1)
    dy = tf.expand_dims(target[:,:,1], -1)
    accpenUp = target[:,:,2]
    Z = ((((dx-mux)/sigmax)**2)+(((dy-muy)/sigmay)**2))-(2*rho*(dx-mux)*(dy-muy)/(sigmax*sigmay))
    N = tf.exp(-Z/(2*(1-rho**2+epsilon))) / (2*3.14159*sigmax*sigmay*(1-rho**2+epsilon)**0.5)
    P = tf.math.log(tf.reduce_sum(pi*N, axis=-1)+epsilon)
    penup = tf.squeeze(penup)
    penloss = -(accpenUp*tf.math.log(penup+epsilon)+(1-accpenUp)*tf.math.log(1-penup+epsilon))
    return tf.reduce_mean(-P+penloss)