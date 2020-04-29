import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from integrated_gradients import integrate_gradients

def create_adv_point(model, ob, alt, base, n_iter, gradient_step=0.01, n_steps=50, delta=1):
    
    ob = tf.convert_to_tensor(ob.reshape((1, ob.shape[0], ob.shape[1])))
    alt = tf.convert_to_tensor(alt.reshape((1, alt.shape[0], alt.shape[1])))
    base = tf.convert_to_tensor(base.reshape((1, base.shape[0], base.shape[1])))
    ob_orig = copy.deepcopy(ob)
    losses = np.zeros((n_iter,))
    for i in range(n_iter):
        gradient, loss = get_adv_gradients(model, ob, alt, base, ob_orig, n_steps, delta)
        ob = ob-(gradient*gradient_step)
        losses[i] = loss
    return ob, losses

def get_adv_gradients(model, ob, alt, base, ob_orig, n_steps=50, delta=1, signed=False):
    """
    model: tf nn model
    ob: observation we want to take the loss with respect to
    alt: alternate point whose explanation we want to mimic
    base: baseline (for Integrated Gradients)
    ob_orig: original observation, pre-perturbation
    n_steps: number of steps for IG
    delta: tradeoff hyperparameter between the explanation 
           difference and prediction difference
    """

    # Gradient tape
    with tf.GradientTape() as tape:    
        # Whatever we watch here, we can get the gradient w.r.t.
        tape.watch(ob)
        
        # Get current adversarial explanation and target explanation
        # (this will be used to "push our explanation away")
        h_x = integrate_gradients(ob, base, model, n_steps)
        h_alt = integrate_gradients(alt, base, model, n_steps)
        
        # Get the model prediction at our current point and
        # where our point was originally
        # (this will keep our adversarial point "close" to the original)
        g_x = model(ob)
        g_x_orig = model(ob_orig)
        
        # Get the explanation difference and prediction difference
        expl_diff = tf.reshape(tf.math.subtract(h_x, h_alt), shape=(1,784))
        pred_diff = tf.math.subtract(g_x, g_x_orig)

        # tf.norm() experiences numerical instability that suffers here, 
        # so we'll just get the norm of our vectors with l2 loss
        expl_norm = tf.map_fn(tf.nn.l2_loss, expl_diff)
        pred_norm = tf.map_fn(tf.nn.l2_loss, pred_diff)
        
        # Convert to double, otherwise addition messes up
        expl_norm = tf.cast(expl_norm, tf.double)
        pred_norm = tf.cast(pred_norm, tf.double)
        
        # Loss is the sum of the explanation difference and prediction difference, 
        # controlled by delta hyperparameter
        loss = expl_norm+pred_norm*delta
    
    # Get gradient of loss function with respect to our observation of interest
    gradient = tape.gradient(loss, ob)
    
    # If only signed gradient is desires, we can return that
    if signed:
        signed_grad = tf.sign(gradient)
        return signed_grad, loss
    # Otherwise return regular gradient (and loss, for tracking convergence)
    return gradient, loss

