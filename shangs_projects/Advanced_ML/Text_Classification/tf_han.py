'''
hierarchical attention network for document classification
https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import sys
import time

class hierarchical_attention_network(object):
    '''
    hierarchical attention network for document classification
    https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num classes: int
        number of output classes
      - max_sents: int
        maximum number of sentences per document
      - max_words: int
        maximum number of words per sentence
      - rnn_type: string (default: "gru")
        rnn cells to use, can be "gru" or "lstm"
      - rnn_units: int (default: 100)
        number of rnn units to use for embedding layers
      - attention_context: int (default: 50)
        number of dimensions to use for attention context layer
      - dropout_keep: float (default: 0.5)
        dropout keep rate for final softmax layer
       
    methods:
      - train(data,labels,epochs=30,savebest=False,filepath=None)
        train network on given data
      - predict(data)
        return the one-hot-encoded predicted labels for given data
      - score(data,labels,bootstrap=False,bs_samples=100)
        return the accuracy of predicted labels on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
    '''
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,rnn_type="gru",
                 rnn_units=200,attention_context=300,dropout_keep=1.0):

        self.rnn_units = rnn_units
        if rnn_type == "gru":
            self.rnn_cell = GRUCell
        elif rnn_type == "lstm":
            self.rnn_cell = LSTMCell
        else:
            raise Exception("rnn_type parameter must be set to gru or lstm")
        self.dropout_keep = dropout_keep
        self.vocab = embedding_matrix
        self.embedding_size = embedding_matrix.shape[1]
        self.ms = max_sents
        self.mw = max_words

        #shared variables
        with tf.variable_scope('words'):
            self.word_atten_W = tf.Variable(self._ortho_weight(2*rnn_units,attention_context),name='word_atten_W')
            self.word_atten_b = tf.Variable(np.asarray(np.zeros(attention_context),dtype=np.float32),name='word_atten_b')
            self.word_softmax = tf.Variable(self._ortho_weight(attention_context,1),name='word_softmax')
        with tf.variable_scope('sentence'):
            self.sent_atten_W = tf.Variable(self._ortho_weight(2*rnn_units,attention_context),name='sent_atten_W')
            self.sent_atten_b = tf.Variable(np.asarray(np.zeros(attention_context),dtype=np.float32),name='sent_atten_b')
            self.sent_softmax = tf.Variable(self._ortho_weight(attention_context,1),name='sent_softmax')
        with tf.variable_scope('classify'):
            self.W_softmax = tf.Variable(self._ortho_weight(rnn_units*2,num_classes),name='W_softmax')
            self.b_softmax = tf.Variable(np.asarray(np.zeros(num_classes),dtype=np.float32),name='b_softmax')
        
        self.embeddings = tf.constant(self.vocab,tf.float32)
        self.dropout = tf.placeholder(tf.float32)

        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_sents,max_words])
        self.words_per_line = tf.reduce_sum(tf.sign(self.doc_input),1)
        self.max_lines = tf.reduce_sum(tf.sign(self.words_per_line))
        self.max_words = tf.reduce_max(self.words_per_line)
        self.doc_input_reduced = self.doc_input[:self.max_lines,:self.max_words]
        self.num_words = self.words_per_line[:self.max_lines]

        #word rnn layer
        self.word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embeddings,dtype=tf.float32),self.doc_input_reduced)
        with tf.variable_scope('words'):
            [word_outputs_fw,word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    tf.contrib.rnn.DropoutWrapper(self.rnn_cell(self.rnn_units),state_keep_prob=self.dropout),
                    tf.contrib.rnn.DropoutWrapper(self.rnn_cell(self.rnn_units),state_keep_prob=self.dropout),
                    self.word_embeds,sequence_length=self.num_words,dtype=tf.float32)
        word_outputs = tf.concat((word_outputs_fw, word_outputs_bw),2)
        
        #word attention
        seq_mask = tf.reshape(tf.sequence_mask(self.num_words,self.max_words),[-1])
        u = tf.nn.tanh(tf.matmul(tf.reshape(word_outputs,[-1,self.rnn_units*2]),self.word_atten_W)+self.word_atten_b)
        exps = tf.exp(tf.matmul(u,self.word_softmax))
        exps = tf.where(seq_mask,exps,tf.ones_like(exps)*0.000000001)
        alpha = tf.reshape(exps,[-1,self.max_words,1])
        alpha /= tf.reshape(tf.reduce_sum(alpha,1),[-1,1,1])
        self.sent_embeds = tf.reduce_sum(word_outputs*alpha,1)
        self.sent_embeds = tf.expand_dims(self.sent_embeds,0)

        #sentence rnn layer
        with tf.variable_scope('sentence'):
            [self.sent_outputs_fw,self.sent_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    tf.contrib.rnn.DropoutWrapper(self.rnn_cell(self.rnn_units),state_keep_prob=self.dropout),
                    tf.contrib.rnn.DropoutWrapper(self.rnn_cell(self.rnn_units),state_keep_prob=self.dropout),
                    self.sent_embeds,sequence_length=tf.expand_dims(self.max_lines,0),dtype=tf.float32)
        self.sent_outputs = tf.concat((tf.squeeze(self.sent_outputs_fw,[0]),tf.squeeze(self.sent_outputs_bw,[0])),1)
        
        #sentence attention
        self.sent_u = tf.nn.tanh(tf.matmul(self.sent_outputs,self.sent_atten_W) + self.sent_atten_b)
        self.sent_exp = tf.exp(tf.matmul(self.sent_u,self.sent_softmax))
        self.sent_atten = self.sent_exp/tf.reduce_sum(self.sent_exp)
        self.doc_embed = tf.transpose(tf.matmul(tf.transpose(self.sent_outputs),self.sent_atten))

        #classification functions
        self.output = tf.matmul(self.doc_embed,self.W_softmax)+self.b_softmax
        self.prediction = tf.nn.softmax(self.output)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(0.00001,0.9,0.99).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
            
    def _ortho_weight(self,fan_in,fan_out):
        '''
        generate orthogonal weight matrix
        '''
        bound = np.sqrt(2./(fan_in+fan_out))
        W = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u
        else:
            W = v
        return W.astype(np.float32)
     
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros((self.ms,self.mw))
            for i,line in enumerate(inputval):
                for j, word in enumerate(line):
                    retval[i,j] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")
     
    def train(self,data,labels,epochs=30,validation_data=None,savebest=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - epochs: int (default: 30)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - savebest: boolean (default: False)
            set to True to save the best model based on validation score per epoch
          - filepath: string (optional)
            path to save model if savebest is set to True
        
        outputs:
            None
        '''
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")
        
        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)
        
        print('training network on %i documents, validating on %i documents' \
              % (len(data), validation_size))
        
        #track best model for saving
        prevbest = 0    
        for i in range(epochs):
            correct = 0.
            start = time.time()
            
            #train
            for doc in range(len(data)):
                inputval = self._list_to_numpy(data[doc])
                feed_dict = {self.doc_input:inputval,self.labels:labels[doc],self.dropout:self.dropout_keep}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],feed_dict=feed_dict)
                if np.argmax(pred) == np.argmax(labels[doc]):
                    correct += 1
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (i+1,doc+1,len(data),cost))
                sys.stdout.flush()
                
                if (doc+1) % 50000 == 0:
                    score = self.score(validation_data[0],validation_data[1])
                    print("iteration %i validation accuracy: %.4f%%" % (doc+1, score*100))
                
            print()
            #print("training time: %.2f" % (time.time()-start))
            trainscore = correct/len(data)
            print("epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100))
            
            #validate
            if validation_data:
                score = self.score(validation_data[0],validation_data[1])
                print("epoch %i validation accuracy: %.4f%%" % (i+1, score*100))
                
            #save if performance better than previous best
            if savebest and score >= prevbest:
                prevbest = score
                self.save(filepath)

    def predict(self,data):
        '''
        return the one-hot-encoded predicted labels for given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
        
        outputs:
            numpy array of one-hot-encoded predicted labels for input data
        '''
        labels = []
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            prob = np.squeeze(prob,0)
            one_hot = np.zeros_like(prob)
            one_hot[np.argmax(prob)] = 1
            labels.append(one_hot)
        
        labels = np.array(labels)
        return labels
        
    def score(self,data,labels):
        '''
        return the accuracy of predicted labels on given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
        
        outputs:
            float representing accuracy of predicted labels on given data
        '''        
        correct = 0.
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct +=1

        accuracy = correct/len(labels)
        return accuracy
        
    def save(self,filename):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load model weights from a file
        
        parameters:
          - filepath: string
            path from which to load model weights
        
        outputs:
            None
        '''
        self.saver.restore(self.sess,filename)
        
if __name__ == "__main__":
        
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.model_selection import train_test_split
    import pickle
    import os

    #load saved files
    print "loading data"
    vocab = np.load('embeddings.npy')
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    num_docs = len(data)

    #convert data to numpy arrays
    print "converting data to arrays"
    max_sents = 0
    max_words = 0
    docs = []
    labels = []
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i+1,num_docs))
        sys.stdout.flush()
        doc = data[i]['idx']
        docs.append(doc)
        labels.append(data[i]['label'])
        if len(doc) > max_sents:
            max_sents = len(doc)
        if len(max(doc,key=len)) > max_words:
            max_words = len(max(doc,key=len))
    del data
    print

    #label encoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = len(le.classes_)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    del labels

    #test train split
    X_train,X_test,y_train,y_test = train_test_split(docs,y_bin,test_size=0.1,
                                    random_state=1234,stratify=y)

    #train nn
    print "building hierarchical attention network"
    nn = hierarchical_attention_network(vocab,classes,max_sents,max_words)
    nn.train(X_train,y_train,epochs=5,validation_data=(X_test,y_test))
