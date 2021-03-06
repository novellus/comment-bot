import os, json, pickle, signal, sys, argparse, functools
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F


class CommentNetwork:
    def __init__(self, n, saveFile, opt, lossFunc, mod=None, use_gpu=False, numDirectIterations=1, defaultOutputTruncation=10):
        self.n=n
        self.saveFile=saveFile
        self.use_gpu=use_gpu
        self.lossFunc=lossFunc
        self.numDirectIterations=numDirectIterations
        self.defaultOutputTruncation=defaultOutputTruncation

        if mod==None:
            #construct network model
            self.model= FunctionSet(
                x_to_h = F.Linear(7, n),
                h_to_h = F.Linear(n, n),
                h_to_y = F.Linear(n, 7)
            )
        else:
            self.model=mod

        if self.use_gpu:
            self.model.to_gpu()
        else:
            self.model.to_cpu()

        self.optimizer = opt
        self.optimizer.setup(self.model)

        #constants
        self.null_byte=np.array([[0]*7], dtype=np.float32)
        if self.use_gpu:
            self.null_byte=cuda.to_gpu(self.null_byte)
        self.null_byte=Variable(self.null_byte)

    def forward_one_step(self, h, x, computeOutput=True):
        h=F.sigmoid(self.model.x_to_h(x) + self.model.h_to_h(h))
        if computeOutput:
            y=F.sigmoid(self.model.h_to_y(h))
            return h, y
        else:
            return h

    def forward(self, input_string, output_string, truncateSize=None, volatile=False):
        if truncateSize==None:
            truncateSize=self.defaultOutputTruncation

        #feed variable in, ignoring output until model has whole input string
        h=np.zeros((1,self.n),dtype=np.float32)
        if self.use_gpu:
            h=cuda.to_gpu(h)
        h=Variable(h, volatile=volatile)
        for c in input_string:
            bits=np.array([[bool(ord(c)&(2**i)) for i in range(7)]], dtype=np.float32)
            if self.use_gpu:
                bits=cuda.to_gpu(bits)
            bits=Variable(bits, volatile=volatile) #8 bits, never all 0 for ascii
            h=self.forward_one_step(h, bits, computeOutput=False)

        #prep for training
        self.optimizer.zero_grads()
        y='' #output string
        nullEnd=False
        loss=0

        def yc_translation(yc, y, nullEnd, truncateSize):
            yc=sum([bool(round(bit))*(2**i_bit) for i_bit, bit in enumerate(cuda.to_cpu(yc.data[0]))]) #translate to int
            if not yc: #null byte signifies end of sequence
                nullEnd=True
            if not nullEnd:
                y+=chr(yc) #translate to character
                truncateSize-=1
            return y, nullEnd, truncateSize

        #Read output by prompting with null bytes.; train with training output
        for c in output_string:
            bits=np.array([[bool(ord(c)&(2**i)) for i in range(7)]], dtype=np.float32)
            if self.use_gpu:
                bits=cuda.to_gpu(bits)
            bits=Variable(bits, volatile=volatile)
            h, yc = self.forward_one_step(h, self.null_byte)
            loss+=self.lossFunc(yc, bits)
            y, nullEnd, truncateSize = yc_translation(yc, y, nullEnd, truncateSize)

        #reinforce null byte as end of sequence
        h, yc = self.forward_one_step(h, self.null_byte) 
        loss+=self.lossFunc(yc, self.null_byte)
        y, nullEnd, truncateSize = yc_translation(yc, y, nullEnd, truncateSize)

        #continue reading out as long as network does not terminate and we have not hit TruncateSize
        while not nullEnd and truncateSize>0:
            h, yc = self.forward_one_step(h, self.null_byte)
            y, nullEnd, truncateSize = yc_translation(yc, y, nullEnd, truncateSize)

        #Train
        loss.backward()
        self.optimizer.update()
        return y, nullEnd #nullEnd true if netowrk terminated output sequence. False if output sequence truncated.

    def trainTree(self, tree, maxCommentLength=float('inf')): #DFS training
        if 'children' in tree:
            allPass=True
            for child in tree['children']:
                self.trainTree(child, maxCommentLength)
                prompt=tree['body']
                trainResponse=child['body']
                if prompt!='[deleted]' and trainResponse!='[deleted]' and prompt and trainResponse and len(prompt)<=maxCommentLength and len(trainResponse)<=maxCommentLength:
                    for i in range(self.numDirectIterations):
                        givenResponse, nullEnd=self.forward(prompt, trainResponse)
                        print '<#'+str(i)+'--prompt--'+str(len(prompt))+'chars-->\n', repr(prompt), '\n<--trainResponse--'+str(len(trainResponse))+'chars-->\n', repr(trainResponse), '\n<--givenResponse--'+str(len(givenResponse))+'chars'+('' if nullEnd else ', truncated')+'-->\n', repr(givenResponse)+'\n'
                        if givenResponse==trainResponse:
                            break
                        else:
                            allPass=False
            return allPass

    # loop over lines in a file identifying if they contain a tree after parsing the json
    def trainFile(self, openFile, maxCommentLength=float('inf')):
        allPass=True
        for i, treeText in enumerate(openFile):
            #throw away whitespace
            if treeText.strip():
                #print fileName, treeText
                tree=json.loads(treeText.strip())
                #it's a tree, let's train
                if 'children' in tree:
                    print 'training #'+str(i)+' '+openFile.name
                    allPass&=self.trainTree(tree, maxCommentLength)
        return allPass

    def saveModel(self):
        print 'Stopped computation, saving model. Please wait...'
        f=open(self.saveFile,'w')
        pickle.dump(self.model, f)
        f.close()
        print 'Saved model'

    def sig_exit(self, _1, _2):
        self.saveModel()
        exit()


# Runs the neural net
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='number of hidden nodes')
    parser.add_argument('-l', metavar='file', type=str, dest='modelFile', help='file to load preexisting model from; n must match model')
    parser.add_argument('--gpu', action='store_const', dest='use_gpu', const=True, default=False, help='Flag to use gpu, omit to use cpu')
    parser.add_argument('-ndi', metavar='#', type=int, default=1, dest='numDirectIterations', help='Num direct iterations before processing next tree, default 1')
    parser.add_argument('-e', metavar='#', type=int, default=1, dest='numEpochs', help='Num epochs, default 1')
    parser.add_argument('-mcl', metavar='#', type=int, default=float('inf'), dest='maxCommentLength', help='Maximum comment length to train with, in characters. Will ignore any comment/response pair where either comment or response are greater than this number. Default is inf.')
    parser.add_argument('-t', metavar='#', type=int, default=10, dest='defaultOutputTruncation', help='Default truncation length for nerual net output. Output will be truncated at max(training response, this value). Defaults to 10 if not specified.')
    parser.add_argument('-i', metavar='file', type=str, dest='treeFile', help='Process only this tree file. If not specified, will process all trees in ./trees')
    parser.add_argument('saveFile', type=str, help='filename to save model to')
    args=parser.parse_args()

    model=None
    #load preexisting model history, if it exists
    if args.modelFile and os.path.exists(args.modelFile): 
        print 'Loading model'
        f=open(args.modelFile)
        model=pickle.load(f)
        f.close()
        print 'Model Loaded'

    #network weight optimizer
    optimizer=optimizers.AdaDelta()
    #optimizer=optimizers.MomentumSGD(momentum=0.9)
    #lossFunc=lambda y1, y2: F.mean_squared_error(y1, y2)
    lossFunc=F.mean_squared_error
    net = CommentNetwork(args.n, args.saveFile, optimizer, lossFunc, model, use_gpu=args.use_gpu, numDirectIterations=args.numDirectIterations, defaultOutputTruncation=args.defaultOutputTruncation)

    #register ctrl-c behavior
    signal.signal(signal.SIGINT, net.sig_exit)

    # go find comment trees to parse
    for epoch in range(args.numEpochs):
        print 'Epoch '+str(epoch)+',',
        allPass=True
        for fileName in map(lambda x:'trees/'+x, os.listdir('trees/')) if not args.treeFile else [args.treeFile]:
            if '.' not in fileName:
                with open(fileName) as f:
                    allPass&=net.trainFile(f, args.maxCommentLength)
        if allPass:
            print 'Breaking on allPass!!'
            break

    print 'Made it through everything, stopping...'
    net.saveModel()
    exit()

if __name__ == "__main__":
    main()
