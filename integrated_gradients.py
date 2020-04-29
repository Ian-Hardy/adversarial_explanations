import pandas as pd
import numpy as np
import tensorflow as tf

"""
Tensorflow implementation of Integrated Gradients from the paper 
"Axiomatic Attribution for Deep Neural Networks"
(link: https://arxiv.org/abs/1703.01365)

TF 2.1.0
"""

def integrate_gradients(ob, base, model, n_steps):
    """
    ob: observation of interest- should be a numpy/tensorflow array, passable to model 
        (kind of a gotcha here- if you have batching enabled, your model input might have an extra dimmension)
    base: baseline- refer to section "2.1. Axiom: Sensitivity(a)" for context. Same requirement as above^
    model: tensorflow model!
    n_steps: number of steps for Riemann integration. Might do some fancier integration in the future, 
        trying to keep it lightweight on the first pass. 
    """
    
    assert len(model.input.shape) == len(ob.shape), \
                                    "Model input and passed observation should have same length shape"
        
    # create path from baseline to observation-- see _paramaterize helper function
    path, step = _parameterize(ob, base, n_steps)
    
    # Really annoying, but because tensorflow doesn't allow for item assignment, 
    # We have to append the gradients to a list and convert to a tensor later. Inefficient! But...
    # Otherwise tf has problems recognizing the gradient of the point w.r.t. integrated gradients.
    # (which we want for adversarial explanations)
    # I tried instantiating a numpy array, doing item assignment, and converting to a tensor after, 
    # but for some reason tensorflow doesn't like that conversion 
    # (Btw, gotta have an extra dimmension to make the model behave because of batching)  
    gradients = []
    
    # Calculate the gradient w.r.t. the input of the model's prediction for every point in the path
    for i, step_point in enumerate(path):
        
        # Weeeeee we love gradient tape
        with tf.GradientTape() as tape:
            
            # Whatever we watch here, we can get the gradient w.r.t.
            tape.watch(step_point)
            
            # Get model prediction at our step point
            prediction = model(step_point)
            
        # Get the gradients of the prediction w.r.t to the input image, assign to gradients[step_i]
        # Also doing an extra step here where we multiply the gradient by the step-- need to for
        # the Riemann sum. Just think about little rectangles haha. 
        gradients.append(tape.gradient(prediction, step_point)*step)
    
    # Convert the list of tensors to a single tensor
    gradients = tf.convert_to_tensor(gradients)
    
    # Sum gradient*step across path-- ADD UP YOUR RECTANGLES NERD
    explanation = tf.reduce_sum(gradients, axis=0)
    
    return explanation


def _parameterize(ob, base, n_steps):
    """
    See above docstring for arg defs
    """
    # Stick with tensorflow math
    # You can make it work with numpy/native python ops, 
    # but if you want to get the gradient w.r.t. the explanation you need tf stuff
    step = tf.math.divide(tf.math.subtract(ob, base), n_steps)
    # No item assignment, so list appending it is
    path = []
    for i in range(n_steps):
        path.append(base+(step*i))
    return path, step
