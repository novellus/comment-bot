import os, json, pickle, signal, sys
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F


class CommentNetwork:
    def __init__(self, n, opt, mod=None):
        self.n=n
        if mod==None:
            #construct network model
            self.model= FunctionSet(
                x_to_h = F.Linear(8, n),
                h_to_h = F.Linear(n, n),
                h_to_y = F.Linear(n, 8)
            )
        else:
            self.model=mod

        self.optimizer = opt
        self.optimizer.setup(self.model.collect_parameters())

        #constants
        self.null_byte=Variable(np.array([[0]*8], dtype=np.float32))

    def forward_one_step(self, h, x):
        h    = F.tanh(self.model.x_to_h(x) + self.model.h_to_h(h))
        y    = self.model.h_to_y(h)
        return h, y

    def forward(self, input_string, output_string, truncateSize=500, volatile=False):
        #feed variable in, ignoring output until model has whole input string
        h=Variable(np.zeros((1,self.n),dtype=np.float32), volatile=volatile)
        for c in input_string:
            bits=Variable(np.array([[bool(ord(c)&(2**i)) for i in range(8)]], dtype=np.float32), volatile=volatile) #8 bits, never all 0 for ascii
            h, _ = self.forward_one_step(h, bits)

        #prep for training
        self.optimizer.zero_grads()
        y='' #output string
        nullEnd=False
        loss=0

        #Read output by prompting with null bytes.; train with training output
        for c in output_string:
            h, yc = self.forward_one_step(h, self.null_byte)
            loss+=F.mean_squared_error(yc, Variable(np.array([[bool(ord(c)&(2**i)) for i in range(8)]], dtype=np.float32)))
            if not any(yc.data[0]): #null byte signifies end of sequence
                nullEnd=True
            if not nullEnd:
                y+=chr(sum([bool(round(bit))*(2**i_bit) for i_bit, bit in enumerate(yc.data[0])]))
                truncateSize-=1

        #reinforce null byte as end of sequence
        h, yc = self.forward_one_step(h, self.null_byte) 
        loss+=F.mean_squared_error(yc, self.null_byte)
        if not any(yc.data[0]):
            nullEnd=True
        if not nullEnd:
            y+=chr(sum([bool(round(bit))*(2**i_bit) for i_bit, bit in enumerate(yc.data[0])]))
            truncateSize-=1

        #continue reading out as long as network does not terminate and we have not hit TruncateSize
        while not nullEnd and truncateSize>0:
            h, yc = self.forward_one_step(h, self.null_byte)
            if not any(yc.data[0]):
                nullEnd=True
            if not nullEnd:
                y+=chr(sum([bool(round(bit))*(2**i_bit) for i_bit, bit in enumerate(yc.data[0])]))
                truncateSize-=1

        #Train
        loss.backward()
        self.optimizer.update()
        return y, nullEnd #nullEnd true if netowrk terminated output sequence. False if output sequence truncated.

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

    def saveModel(self):
        print 'Stopped computation, saving model. Please wait...'
        f=open('model','w')
        pickle.dump(self.model, f)
        f.close()
        print 'Saved model'

    def sig_exit(self, _1, _2):
        self.saveModel()
        exit()


# Runs the neural net
def main():

    cuda.init()

    model=None
    #load preexisting model history, if it exists
    if os.path.exists('model'): 
        print 'Loading model'
        f=open('model')
        model=pickle.load(f)
        f.close()
        print 'Model Loaded'

    #network weight optimizer
    optimizer=optimizers.MomentumSGD()
    net = CommentNetwork(16000, optimizer, model)

    #register ctrl-c behavior
    signal.signal(signal.SIGINT, net.sig_exit)

    # go find comment trees to parse
    for fileName in os.listdir('trees/'):
        if '.' not in fileName:
            with open('trees/'+fileName) as f:
                net.trainFile(f)

    print 'Made it through everything, stopping...'
    net.saveModel()
    exit()

if __name__ == "__main__":
    main()
