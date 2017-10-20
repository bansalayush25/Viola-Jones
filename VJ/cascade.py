import numpy as np
from functools import partial

def ensemble(img, classifiers):
    return 1 if sum([c.get_vote(img) for c in classifiers])>=0 else 0

def ensemble_all(images, classifiers):
    votes_partial = partial(ensemble, classifiers=classifiers)
    return list(map(votes_partial, images))