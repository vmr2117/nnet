import numpy as np
from vowpal_porpoise import VW
import sys
import os
import tempfile
import commands
import itertools

class weak_learner:
    def __init__(self,  moniker, cache_name, passes):
        self.moniker=moniker
        self.model=VW(moniker=moniker, name= cache_name, passes=passes , csoaa=10)


    def train(self, instance_stream):
        with self.model.training():
            seen=0
            for instance in instance_stream:
                self.model.push_instance(instance)
                seen+=1
                if seen % 1000 ==0:
                    print 'setreamed %d instances already' % seen
            print 'streaming finished'
        print '%s: trained on %d data points' % (self.moniker, seen)

    def predict(self, instance_stream):
        #print '%s: predicting' % self.moniker
        instances = []
        seen=0
        
        with self.model.predicting():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                instances.append(instance)
                seen += 1
        print '%s: predicted for %d data points' % (self.moniker, seen)
        predictions = list(self.model.read_predictions_())
        if seen != len(predictions):
           raise Exception("Number of labels and predictions do not match!  (%d vs %d)" % \
                            (seen, len(predictions)))
        return  predictions[:len(predictions)]


