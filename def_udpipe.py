# This file is part of UDPipe <http://github.com/ufal/udpipe/>.
#
# Copyright 2016 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import ufal.udpipe


class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output

model = Model('czech-pdt-ud-2.4-190531.udpipe')

'''
#UDPipe preprocessing
def UDPipe_preprocessing_sentence(text):
    
    sentences = model.tokenize(text)
    
    for s in sentences:
        model.tag(s)
        model.parse(s)
    
    conll = model.write(sentences, "conllu")
    conll = conll.split('\n')
    result = []
    
    
    for i in range(len(conll)):
        tmp_sentence = []
        for line in conll:
            
            if line.startswith("#") or len(line) == 0:
                               continue
            
            splitted = line.split()
            #extract lemma 
            tmp_sentence.append(splitted[2])
            
        tmp_result =  ' '.join(tmp_sentence)
    result.append(tmp_result)
        
    return result
'''


#UDPipe preprocessing
def UDPipe_preprocessing_word(text):
    
    sentences = model.tokenize(text)
    
    for s in sentences:
        model.tag(s)
        model.parse(s)
    
    conll = model.write(sentences, "conllu")
    conll = conll.split('\n')
    result = ""
    
    for line in conll:    
        if line.startswith("#") or len(line) == 0:
                           continue
        
        splitted = line.split()
        #extract lemma 
        result += splitted[2] + " "
            
    return result


