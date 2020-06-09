# Requirements
### Minimal python version
* 3.7.4

### Python libraries
* ufal.udpipe
* stop_words
* re
* pandas 
* gensim
* smart_open
* json 

# How to run
1. open cmd
1. `git clone https://github.com/HoaVuThu/diplomka.git`
1. `cd diplomka`
1. Download [model](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/czech-pdt-ud-2.4-190531.udpipe?sequence=20&isAllowed=y) for tokenization and lemmatization into diplomka folder
1. `python cmdParamReadingExample.py <string> <string>`

# Example
**command:**

`python cmdParamReadingExample.py "infarkt" "srdeční nemoci"`

**output:**
```
First Argument: infarkt
Second Argument: srdeční nemoc
Cosinus similarity: 0.8432024121284485
Cosinus similarity ranged from 0 to 1: 92.16
```
