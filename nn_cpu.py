import os, json, pickle, signal, sys
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

# predefined globals
n=16000
net = None

class CommentNetwork:
    def __init__(self, mod, opt):
        self.model = mod
        self.optimizer = opt

    def forward_one_step(self, h, x):
        h    = F.tanh(self.model.x_to_h(x) + self.model.h_to_h(h))
        y    = self.model.h_to_y(h)
        return h, y

    def forward(self, input_string, output_string, volatile=False):
        h=Variable(np.zeros((1,n),dtype=np.float32), volatile=volatile)
        for c in input_string:
            bits=Variable(np.array([[bool(ord(c)&(2**i)) for i in range(8)]], dtype=np.float32), volatile=volatile) #8 bits, never all 0 for ascii
            h, _ = self.forward_one_step(h, bits)

        self.optimizer.zero_grads()
        y=''
        nullEnd=False
        loss=0

        for c in output_string:
            h, yc = self.forward_one_step(h, Variable(np.array([[0]*8], dtype=np.float32), volatile=volatile))
            loss+=F.mean_squared_error(yc, Variable(np.array([[bool(ord(c)&(2**i)) for i in range(8)]], dtype=np.float32)))
            if not any(yc.data[0]):
                nullEnd=True
            if not nullEnd:
                y+=chr(sum([bool(round(bit))*(2**i_bit) for i_bit, bit in enumerate(yc.data[0])]))
        _, yc = self.forward_one_step(h, Variable(np.array([[0]*8], dtype=np.float32),  volatile=volatile))
        loss+=F.mean_squared_error(yc, Variable(np.array([[0]*8], dtype=np.float32)))

        loss.backward()
        self.optimizer.update()
        return y, nullEnd

    def trainTree(self, tree): #DFS training
        if 'children' in tree:
            for child in tree['children']:
                self.trainTree(child)
                prompt=tree['body']
                trainResponse=child['body']
                if prompt!='[deleted]' and trainResponse!='[deleted]' and prompt and trainResponse:
                    self.optimizer.zero_grads()
                    givenResponse, nullEnd=self.forward(prompt, trainResponse)
                    print '<--prompt--'+str(len(prompt))+'chars-->\n', prompt, '\n<--trainResponse--'+str(len(trainResponse))+'chars-->\n', trainResponse, '\n<--givenResponse--'+str(len(givenResponse))+'chars'+('' if nullEnd else ', truncated')+'-->\n', givenResponse, '\n\n'

    # loop over lines in a file identifying if they contain a tree after parsing the json
    def trainFile(self, openFile):
        for treeText in openFile:
            #throw away whitespace
            if treeText.strip():
                #print fileName, treeText
                tree=json.loads(treeText.strip())
                #it's a tree, let's train
                if 'children' in tree:
                    print 'training '+openFile.name+'\n'
                    self.trainTree(tree)

    def saveModel():
        print 'Stopped computation, saving model. Please wait...'
        f=open('model','w')
        pickle.dump(self.model, f)
        f.close()
        print 'Saved model'

def sig_exit(_1, _2):
    saveModel()


# Runs the neural net
def main():

    cuda.init()

    model=None
    if os.path.exists('model'):
        #load preexisting model history
        print 'Loading model'
        f=open('model')
        model=pickle.load(f)
        f.close()
        print 'Model Loaded'
    else:
        #construct network model
        model= FunctionSet(
            x_to_h = F.Linear(8, n),
            h_to_h = F.Linear(n, n),
            h_to_y = F.Linear(n, 8)
        )

    #network weight optimizer
    optimizer=optimizers.MomentumSGD()
    optimizer.setup(model.collect_parameters())

    net = CommentNetwork(model, optimizer)

    #register ctrl-c behavior
    signal.signal(signal.SIGINT, sig_exit)

    # go find comment trees to parse
    for fileName in os.listdir('trees/'):
        if '.' not in fileName:
            with open('trees/'+fileName) as f:
                net.trainFile(f)

    print 'Made it through everything, stopping...'
    saveAndQuit()

if __name__ == "__main__":
    main()
