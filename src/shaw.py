#!/bin/python3
'''
Runs standard input through a part-of-speech tagger, then
translates to Shavian. This resolves most heteronyms, but
do still check the output for @ signs and fix them by hand.

Each line of a dictionary consists of an English word, a space,
a Shavian translation, and no comments. Special notations are:

^word 𐑢𐑻𐑛	word is a prefix
$word 𐑢𐑻𐑛	word is a suffix
Word 𐑢𐑻𐑛	always use a namer dot
word_ 𐑢𐑻𐑛	never use a namer dot
word_VB 𐑢𐑻𐑛	shave this way when tagged as a verb
word. 𐑢𐑻𐑛	shave this way when no suffix is present
word .𐑢𐑻𐑛	word takes no prefixes
word 𐑢𐑻𐑛.	word takes no suffixes
word 𐑢𐑻𐑛:	suffixes do not alter the root,
	      	e.g. "𐑑𐑾" or "𐑕𐑾" palatizing to "𐑖𐑩" or "𐑠𐑩".
word .		delete this word from the dictionary

Words are matched case-sensitive when possible, e.g. US/us,
WHO/who, Job/job, Nice/nice, Polish/polish.

shaw.py does not care about the order of dictionary entries.
shaw.c requries a highly specific order not described here.
'''

import re
import os
import sys
import html
from html.parser import HTMLParser

apostrophe = "'"	# whatever you want for apostrophe, e.g. "’" or ""
merge_ia = True		# True: 𐑣𐑨𐑐𐑽, 𐑣𐑨𐑐𐑾𐑕𐑑  False: 𐑣𐑨𐑐𐑦𐑼, 𐑣𐑨𐑐𐑦𐑩𐑕𐑑
runic_vee = "ᚡ"		# could use ᚠ, ᚹ, ᚢ, or ᚠ\u200dᚠ
dot_entire_name = True
if os.path.exists('config.py'):
  from config import *

script = postag = alphabet = False
dict = htags = { }
tokens = [ "." ]
units = { "ms":"𐑥𐑕","bc":"𐑚𐑰𐑕𐑰","psi":"𐑐𐑕𐑦","pc":"𐑐𐑕","mi":"𐑥𐑲" }
contr = [ "'d","'ll","'m","n't","'re","'s","'ve" ]
abbrev = [ "abbr","acad","al","alt","apr","assn","at","aug","ave","b","c","ca","cf",
	"capt","cent","chm","chmn","co","col","comdr","corp","cpl","d","dec","dept","dist",
	"div","dr","ed","esq","est","etc","feb","fl","gen", "gov","hon","inc",
	"inst","jan","jr","lat","lib","lt","ltd","mar","mr","mrs","ms","msgr",
	"mt","mts","mus","nov","oct","pg","phd","pl","pop","pp","prof","pseud","pt",
	"rev","sept","ser","sgt","sr","st","uninc","univ","vol","vs","wt" ]

# Remove diacritics from Latin letters, break up ligatures, and do nothing else.
def unaccent (str):
  map = "AAAAAA CEEEEIIIIDNOOOOO OUUUUY  aaaaaa ceeeeiiiidnooooo ouuuuy yAaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIi  JjKkkLlLlLlLlLlNnNnNn   OoOoOo  RrRrRrSsSsSsSsTtTtTtUuUuUuUuUuUuWwYyYZzZzZz bBBb   CcDDDdd  EFfG  IIKkl  NnOOo  Pp     tTtTUuYVYyZz    255      Ǳǲǳ      AaIiOoUuUuUuUuUu AaAaÆæGgGgKkOoOo  j   Gg  NnAaÆæOoAaAaEeEeIiIiOoOoRrRrUuUuSsTt  HhNd  ZzAaEeOoOoOoOoYylntj  ACcLTsz  BU EeJjqqRrYy"
  ext = "AaBbBbBbCcDdDdDdDdDdEeEeEeEeEeFfGgHhHhHhHhHhIiIiKkKkKkLlLlLlLlMmMmMmNnNnNnNnOoOoOoOoPpPpRrRrRrRrSsSsSsSsSsTtTtTtTtUuUuUuUuUuVvVvWwWwWwWwWwXxXxYyZzZzZzhtwyasssSSẟAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYyLLllVvYy"
  lig = { 'Æ':'AE','æ':'ae','Ǳ':'DZ','ǲ':'Dz','ǳ':'dz','Ĳ':'Ij','ĳ':'ij','Ǉ':'LJ','ǈ':'Lj','ǉ':'lj','Ǌ':'NJ','ǋ':'Nj','ǌ':'nj','Œ':'OE','œ':'oe','Ƣ':'OI','ƣ':'oi','ß':'ss','ﬀ':'ff','ﬁ':'fi','ﬂ':'fl','ﬃ':'ffi','ﬄ':'ffl','ﬅ':'st','ﬆ':'st' }
  ret = ""
  for char in str:
    n = ord(char)
    if n >= 0xc0 and n < 0x250 and map[n-0xc0] != ' ': char = map[n-0xc0]
    if n >= 0x1e00 and n < 0x1f00: char = ext[n-0x1e00]
    if n >= 0x300 and n < 0x370: char = ""
    if char in lig: char = lig[char]
    ret += char
  return ret

def notrans (str):
  global htags
  if toki in htags:
    htags[toki] += str
  else:
    htags[toki] = str

def tokenize (skr):
  global tokens, toki
  if not isinstance(skr,str): skr = ""
  skr = " "+unaccent(html.unescape(skr))+" "
  old = 0
  for i in range(1,len(skr)-1):
    new = 0
    if skr[i].isalpha(): new = 1
    if skr[i].isdigit(): new = 2
    if skr[i] == " " and tokens[-1][0].isalpha() and skr[i+1].isalpha(): new = 0
    if skr[i] == "'"  and skr[i-1].isalpha() and skr[i+1].isalpha(): new = 1
    if skr[i] in ",." and skr[i-1].isdigit() and skr[i+1].isdigit(): new = 2
    if skr[i] == "." and new == 0 and tokens[-1].lower() in abbrev: continue
    if old and old == new: tokens[-1] += skr[i]
    else:
      for c in contr:			# break up contractions so PoS tagging works
        s = len(tokens[-1]) - len(c)
        if s < 1: continue
        low = tokens[-1][s:].lower()
        if c == low:
          tokens[-1] = tokens[-1][:s]
          tokens.append(low)
      tokens.append(skr[i])
      toki += 1
    old = new
    if tokens[-1].isspace() or not tokens[-1].isprintable() or ord(tokens[-1][0]) | 15 == 0xfe0f:
      toki -= 1				# Whitespace tokens break NLTK and variation
      notrans (tokens.pop())		# selectors break Flair. Move these to htags.

class MyHTMLParser(HTMLParser):
  def handle_starttag (self, tag, attrs):
    global script
    out = '<'+tag
    for at in attrs:
      if at[0] == 'charset':
        at = ('charset', 'UTF-8')
      if at[0] == 'content':
        at = ('content', 'text/html; charset=UTF-8')
      out += ' '+at[0]
      if at[0] in ('alt','title'):
        notrans (out+'="')
        tokenize (at[1])
        out = '"'
      elif type(at[1]) == str:
        out += '="'+at[1]+'"'
    out += '>'
    if tag == 'noscript' or tag == 'script' or tag == 'style':
      script = True
    notrans (out)

  def handle_endtag (self, tag):
    global script
    notrans ('</'+tag+'>')
    if tag == 'noscript' or tag == 'script' or tag == 'style':
      script = False

  def handle_data (self, data):
    if script: notrans (data);
    else:     tokenize (data);

# Search all the ways a word might appear in the dictionary
def lookup (word, pos):
  ret = ""
  low = word.lower()
  upp = word[0].upper()+low[1:]
  pos = "_" + pos
  if aflag & 2: list = [ low+pos, low+pos[:3], word, word+"_",low, low+"_", low+"_NN", low+"_NNS", upp ]
  else: list = [ low+pos,low+pos[:3],word+".",upp+".",word,word+"_",low+".",low,low+"_",low+"_NN",low+"_NNS",upp ]
  for look in list:
    if look in dict:
      ret = dict[look]
      if aflag & 1 and ret[0] == '.' or aflag & 2 and ret[-1] == '.': ret = ""
      ret = ret.replace(".","")
      if not ret: continue
      if (word[0].isupper() or look[0].isupper()) \
	   and (look[-1] != "_" or aflag) and not re.search("[A-Z]",ret):
        ret = "·" + ret
      break
  return ret

def suffix_split (inp, pos, adj):
  global aflag
  long = len(inp)
  root = lookup (inp, pos)
  if (root): return ((long+adj)**2, root)
  low = inp.lower();
  best = (0, "")
  for split in range(2,long):
    suff = "$"+low[split:]
    if not suff in dict: continue
    if low[split] == low[split-1]:
      if long-split == 1 or low[split] in "eos": continue
    if low[split-1] == low[split-2]:
      if low[split:] == "ess" and low[split-1] in "ln": continue
    else:
      if low[split:] == "ry" and low[split-1] in "aeiouf": continue
      if low[split:] == "ha" and low[split-1] in "cpst": continue
      if low[split:] == "th" and low[split-1] in "e": continue
      if low[split:] == "d" and low[split-1] in "adeiou": continue
      if low[split:] == "w" and low[split-1] in "aeo": continue
      if low[split:] == "t" and low[split-1] in "aeioust": continue
      if low[split:] == "k" and low[split-1] in "aceino": continue
      if low[split:] == "r" and low[split-1] in "aeiou": continue
      if low[split:] == "m" and low[split-1] in "eis": continue
      if low[split:] == "z" and low[split-1] in "i": continue
      if low[split:] == "n" and low[split-1] in "eio": continue
    if low[split:] == "es" and low[split-1] not in "hiosuxz": continue
    suff = dict[suff]
    if aflag & 2 and suff[-1] == '.': continue
    for pess in range(2):
      if pess:
        word = inp[:split]
      elif low[split-1] == 'i' and low[split] not in "cfikmpsv":
        word = inp[:split-1] + 'y'
      elif low[split] in "aeiouy'" and low[split-1] not in "aeio" \
		and low[split:split+2] not in [ "ub","up" ]:
        if low[split-1] == low[split-2] and low[split-1] not in "hsw":
          word = inp[:split-1]
        elif ( low[split-1] in "cdghlsuvz" or low[split] == 'e' or low[split-2] in "aeiousy" ) \
		and (low[split-1] not in "cg" or low[split] not in "aou"):
          word = inp[:split] + "e"
        else: continue
      elif low[split-2:split] == "dg":
        word = inp[:split] + "e"
      else: continue
      sflag = aflag
      aflag &= ~2
      if inp[split] != "'" or word != inp[:split]: aflag |= 2
      root = suffix_split (word, "UNK", split-len(word))
      score = (long-split+adj)**2 + root[0] if root[0] else 0
      aflag = sflag
      if score:
        if low[split] == "'" and pess == 0: score /= 2
        if low[split:] in [ "call" ]: score = 1
        if low[split:] in [ "bed","can","cat","cent","dance","ine","kin","one","pal","path","ster","tie","tied","ties","tying","wing","x" ]:
          score = max (1, score - 9)
      if (score <= best[0]): continue
      root = root[1]
      if low[split-1] == 'e' and low[split-2] not in "aegiou" and low[split] in "aou" \
	  and (split+1 == long or low[split+1] in "dlmnprstu") \
	  and low[split:] not in [ "arm","out","und","up" ]:
        if root[-1] in "𐑦𐑰": root = root[:-1]
        root += "𐑦"
      if root[-1] == "𐑓" and suff[0] > "𐑗" and word[-1] in "VWvw" and low[split:] != "s":
        root = root[:-1] + "𐑝"
      if root[-2:] == "𐑩𐑤" and suff == "𐑦" and word[-2:].lower() == "le":
        root = root[:-2] + "𐑤"
      if root[-3:] in ["𐑞𐑩𐑥","𐑟𐑩𐑥"] and suff not in ["'","𐑛:","𐑟:","𐑦𐑙"]:
        root = root[:-2] + "𐑥"
      if root[-2:] in ["𐑩𐑤","𐑭𐑤","𐑾𐑤"] and suff == "𐑦𐑑𐑦":
        mid = "𐑦" if root[-3] in "𐑖𐑗𐑠𐑡" or root[-2] == "𐑾" else ""
        root = root[:-2] + mid + "𐑨𐑤"
      mid = root[-1] + suff[0]
      if merge_ia:
        if mid == "𐑦𐑩": mid = "𐑾"
        if mid == "𐑦𐑼": mid = "𐑽"
      if mid in ["𐑤𐑤","𐑯𐑯"] and len(suff) < 3: mid = mid[0]
      best = (score, root[:-1] + mid + suff[1:])
  if long > 1 and low[-1] == low[-2] and low[-2] not in "aeiosu":
    aflag |= 2
    root = suffix_split (inp[:-1], "UNK", 0)
    if best[0] < root[0]: best = root
  if len(best[1]) > 1:
    word = best[1][:-2]
    end  = best[1][-2:]
    if end in [ "𐑛:","𐑟:" ]:
      tail = -1
      while word[tail] in ["'",":"]: tail -= 1
      if word[tail] in {"𐑛:":"𐑑𐑛","𐑟:":"𐑕𐑖𐑗𐑟𐑠𐑡"}[end]:
        word += "𐑩"
      elif word[tail] >= "𐑐" and word[tail] < "𐑘":
        end = chr(ord(end[0])-10)+":"
    word += end
    if word[-4:] == "𐑒𐑩𐑤𐑦" and word[-5] in "𐑦𐑩":
      word = word[:-4] + "𐑒𐑤𐑦"
    if word[-4:-2] == "𐑑𐑵" and word[-2] in "𐑩𐑱𐑺𐑼":
      word = word[:-4] + "𐑗" + word[-3:]
    pal = word[::-1].replace("𐑩𐑦","𐑾",1)[::-1].split("𐑾")
    if len(pal) > 1 and len(pal[-2]) > 1 and \
	pal[-2][-1] in "𐑑𐑕𐑟" and pal[-1] in ["𐑕","𐑤","𐑯",""]:
      mid = "𐑖"
      if pal[-2][-1] == "𐑑":
        if pal[-2][-2] == "𐑕": mid = "𐑗"
      elif pal[-1] in ["𐑯",""] and pal[-2][-2] in "𐑰𐑱𐑴𐑵𐑷𐑻𐑿": mid = "𐑠"
      pal[-2] = pal[-2][:-1] + mid + "𐑩" + pal[-1]
      word = "𐑾".join(pal[:-1])
    best = (best[0], word)
  return best

def prefix_split (word, pos, ms):
  global aflag
  best = suffix_split (word, pos, 0)
  if best[0] == len(word)**2: return best
  for split in range(len(word)-2,ms,-1):
    pref = "^"+word[:split].lower()
    if not pref in dict: continue
    if word[:split+1].lower() == "un": continue
    if pref == "^z" and word[split].lower() in "aeiouy": continue
    aflag = word[split-1] != "'"
    root = prefix_split (word[split:], pos, 1)
    score = split**2 + root[0] if root[0] else 0
    if pref == "^la": score -= 4;
    if score > best[0]:
      dot = "·" if word[0].isupper() else ""
      pref = dict[pref]
      init = root[1][0]
      if (init == "·"): init = root[1][1]
      if pref[-1] == init and pref[-1] in "𐑤𐑥𐑮𐑯" and pref[-2] in '𐑦𐑧' \
	or pref == "𐑥𐑩𐑒" and init == "𐑒":
        pref = pref[:-1]
      best = (score, pref + dot + root[1])
  return best

if len(sys.argv) < 2:
  print ("Usage:",sys.argv[0],"file1.dict file2.dict ...")
  exit()

first = True
for fname in sys.argv[1:]:
  if postag == -1:
     postag = int(fname)
     continue
  if alphabet == -1:
     alphabet = int(fname)
     continue
  if fname == "-p":		# Choose part-of-speech tagger, default none
    postag = -1
    continue
  if fname == "-a":		# Choose output alphabet, default Shavian
    alphabet = -1
    continue
  with open (fname, 'r', encoding="utf-8") as df:
    for line in df:
      word = line.split()
      if alphabet == 8 and word[0][-1] in "vw" and word[1][-1] == "𐑓":
        word[1] = word[1][:-1] + "𐑝"
      if not postag:
        word[0] = re.sub('_[A-Z]+','',word[0])
        if word[0]+'_' in dict: word[0] += '_'
      if first and word[0] in dict:
        if word[1] not in dict[word[0]].split('@'):
          dict[word[0]] += "@"+word[1]
      else:
        dict[word[0]] = word[1]
      if word[1] == ".":
        del dict[word[0]]
      elif not first:		# Allow extra dicts to force dotting
        low = word[0].lower()
        if low != word[0] and low in dict: del dict[low]
  first = False

if alphabet: merge_ia = False
if merge_ia:
  for word in dict:
    dict[word] = dict[word].replace("𐑦𐑩","𐑾").replace("𐑦𐑼","𐑽")

text = sys.stdin.read()
# Any morpheme containing periods, hyphens, or slashes needs special treatment here
for tup in [("([ʻˈ‘’ʼ´`ʿ]|&(#8217|rsquo);)","'"),("[\u00ad\u200b]",""),
	(r'\be\.g\.','igz'),(r'\bi\.e\.','ie'),(r'\bph\.d\.','phd'),
	(r'\[([a-z])\]',r'\1'),(r'\bde-','dee-'),(r'\bt-','tee-'),("'tee-","'t-"),
	(r'\bw/o\b','without'),(r'\bw/','with '),('vis-[aà]-vis','vis-ah-vee'),
	('sine qua','sinna qua')]:
  text = re.sub(tup[0],tup[1],text,count=0,flags=re.I)

toki = 1
parser = MyHTMLParser()
parser.feed (text)

tags = []
if postag == 1:			# Do part-of-speech tagging
  import nltk
  tags = nltk.pos_tag (tokens)
elif postag == 2:
  import spacy
  spaCy = spacy.load("en_core_web_sm")
  tags = [(w.text, w.tag_) for w in spaCy(spacy.tokens.Doc(spaCy.vocab, tokens))]
elif postag == 3:
  save = sys.stdout
  sys.stdout = sys.stderr
  from flair.data import Sentence
  from flair.models import SequenceTagger
  flair = SequenceTagger.load("flair/pos-english-fast")
  sys.stdout = save
  sen = []
  for tok in tokens + ['.']:
    sen.append(tok)
    if tok in ['.','?','!']:
      sen = Sentence(sen)
      flair.predict(sen)
      for tok in sen:
        tags.append((tok.text, tok.get_label('pos').value))
      sen = []
  del tags[-1]
else:
  tags = [(tok,"UNK") for tok in tokens]

jtags = []			# Re-join broken contractions
for token in tags:
  if token[0].lower() in contr:
    jtags[-1] = (jtags[-1][0]+token[0],jtags[-1][1]+"+"+token[1])
  else: jtags.append(token)

out = []			# Translate to Shavian
prev = (".",'.')
toki = 1
initial = maydot = True
map = { "𐑑":"𐑑𐑩","𐑓":"𐑓𐑹","𐑝":"𐑩𐑝","𐑯":"𐑩𐑯𐑛","𐑞":"𐑞𐑩" }
if alphabet == 1:
  map = { "𐑑":"𐑑𐑵","𐑓":"𐑓𐑹","𐑝":"𐑪𐑝","𐑯":"𐑨𐑯𐑛","𐑩𐑯":"𐑨𐑯","𐑩":"𐑭" }
if alphabet == 3:
  apostrophe = "ᛌ"
  map.update({"𐑘𐑧𐑩":"𐑘𐑧"})
if alphabet == 6:
  map = { "𐑑":"𐑑𐑵","𐑓":"𐑓𐑹","𐑝":"𐑪𐑝","𐑯":"𐑨𐑯𐑛","𐑞":"𐑞𐑦","𐑩𐑯":"𐑨𐑯","𐑩":"𐑧" }
if alphabet == 7:
  map = { "𐑦𐑑":"𐑦","𐑦𐑟":"𐑟","𐑚𐑰":"𐑚" }
for token in jtags[1:]:
  if toki in htags:
    out.append(htags[toki])
    if htags[toki].lower().find("<div")+1:
      initial = True
  toki += 1
#  print (token, file=sys.stderr)
  if token[0] in ['.','?','!',':','"','“','”']:
    initial = True
  word = token[0]
  low = word.lower()
  befto = { "have":"𐑨𐑓","has":"𐑨𐑕","used":"𐑕𐑑","unused":"𐑕𐑑","supposed":"𐑕𐑑"}
  if prev[0] in befto and low == "to":
    if not postag: out.extend(["@",befto[prev[0]]," "])
    elif prev != ("used","VBN"):	# If "to" changes the meaning of the preceding
      for i in reversed(range(len(out))): # word, it also changes the pronunciation.
        if out[i] == tran: break
      out[i] = tran[:-2] + befto[prev[0]]
  tran = word
  if word.split("'")[0] == "I": word = "i" + word[1:]
  if word[0].isalpha() and not re.fullmatch("[WMDC]*[CLX]*[XVI]*('(s|d|ll))?",word):
    dot = word[0].isupper()
    if initial:
      initial = False
      if word.split("'")[0] == "i": dot = True
      if len(word) == 1 or word[1].islower() and word not in ["God"]:
        word = word[0].lower() + word[1:]
    aflag = 0
    root = prefix_split (word, token[1], 0)
    if root[1]: tran = root[1].replace("'",apostrophe).replace(":","").replace(".","")
    if len(tran) > 2 and tran[-2:] == "𐑥𐑚": tran = tran[:-1]
    if prev[0][0].isdigit() and low in units:
      tran = units[low]
    if alphabet in [0,7,10]: dot = maydot and "·" in tran
    tran = tran.replace("·","").replace("𐑲𐑟𐑱𐑖𐑩𐑯",dict["ization"][1:])
    if alphabet:
      if tran in map: tran = map[tran]
      if low not in [ "his","tis" ] and low.split("i")[-1] not in [ "d","dd","dde","z","zz" ]:
        for i in range(1,len(tran)):
          if tran[i] == "𐑦" and (tran[i+1:] in [ "","𐑛","𐑟" ] or tran[i+1] in "𐑦𐑧𐑨𐑩𐑪𐑫𐑬𐑭𐑰𐑱𐑲𐑳𐑴𐑵𐑶𐑷𐑸𐑹𐑺𐑻𐑼"):
            tran = tran[:i] + "𐒀" + tran[i+1:]
      for rep in ['𐒁𐑣𐑒','𐒂𐑣𐑜','𐒃𐑣𐑤','𐒄𐑣𐑢','𐒅𐑺︀','𐒆𐑻︀','𐒁𐑒︀','𐒂𐑜︀','𐒃𐑤︀','𐒄𐑢︀']: # Those without 𐑣 contain VS1
        tran = tran.replace(rep[1:],rep[0])
    if alphabet == 3:
      for tup in [("([𐑬𐑱𐑲𐑴𐑵𐑶𐑿])𐑼","\\1𐑮"),("𐑫𐑼","𐑫\u200d𐑮"),("𐑒𐑢","ᛢ"),("𐑕𐑗","𐑕𐑑𐑘"),("𐑕𐑑","ᛥ")]:
        tran = re.sub(tup[0],tup[1],tran)
    if alphabet == 5:
      for tup in [("([𐑧𐑨𐑪𐑳])𐑮","\\1𐑮𐑮"),("([𐑑𐑒𐑕𐑛𐑜𐑟])𐑣","\\1‧𐑣"),("𐑯𐑜","𐑯‧𐑜"),("𐑤𐑤","𐑤‧𐑤")]:
        tran = re.sub(tup[0],tup[1],tran)
    if alphabet == 6:
      tran = re.sub("([𐑦𐑧𐑨𐑩𐑪𐑫𐑬𐑭𐑰𐑱𐑲𐑳𐑴𐑵𐑶𐑷])𐑼","\\1𐑮",tran)
    if alphabet == 7:
      if "x" in low: tran = tran.replace("𐑒𐑕","",1).replace("𐑜𐑟","",1)
    if dot:
      tran = "·" + tran
      maydot = dot_entire_name
#      if token[0][0].islower(): print ("DOT",token[0])
    else:
      maydot = True
#      if token[0][0].isupper(): print ("NODOT",token[0])
  elif word != '-': maydot = True	# Names may contain hyphens
  out.append(tran)
  if low != " ": prev = (low,token[1])
if toki in htags:
  out.append(htags[toki])
out = "".join(out)
if alphabet:			# Translate Shavian to something else
  letters = [
      [ "𐐹","𐐻","𐐿","𐑁","𐑃","𐑅","𐑇","𐐽","𐐷","𐑍",	# Deseret
	"𐐺","𐐼","𐑀","𐑂","𐑄","𐑆","𐑈","𐐾","𐐶","𐐸",
	"𐑊","𐑋","𐐮","𐐯","𐐰","ɪ","𐐱","𐐳","𐐵","𐐪",
	"𐑉","𐑌","𐐨","𐐩","𐐴","𐐲","𐐬","𐐭","𐑎","𐐫",
	"𐐪𐑉","𐐫𐑉","𐐩𐑉","𐐲𐑉","ɪ𐑉","𐐨𐑉","𐐨ɪ","𐑏",
	"𐐮","𐐸𐐿","𐐸𐑀","𐐸𐑊","𐐸𐐶","𐐩","𐐲" ],
      [ "p","t","k","f","θ","s","ʃ","tʃ","j","ŋ",	# IPA
	"b","d","g","v","ð","z","ʒ","dʒ","w","h",
	"l","m","ɪ","ɛ","æ","ə","ɒ","ʊ","aʊ","ɑː",
	"r","n","iː","eɪ","aɪ","ʌ","əʊ","uː","ɔɪ","ɔː",
	"ɑːr","ɔːr","eər","ɜːr","ər","ɪər","ɪə","juː",
	"i","x","ɣ","ɬ","hw","eə","ɜː" ],
      [ "ᛈ","ᛏ","ᛣ","ᚠ","ᚦ","ᛋ","ᛋᚳ","ᚳ","ᛄ","ᛝ",	# Runic
	"ᛒ","ᛞ","ᚸ",runic_vee,"ᚦ","ᛉ","ᛉᚳ","ᚷ","ᚹ","ᚻ",
	"ᛚ","ᛗ","ᛁ","ᛖ","ᚫ","ᚣ","ᚩ","ᚢ","ᚫᚢ","ᚪ",
	"ᚱ","ᚾ","ᛇ","ᛖᛡ","ᚪᛡ","ᚣ","ᛟ","ᚢ","ᛟᛡ","ᚩ",
	"ᚪᚱ","ᚩ\u200dᚱ","ᛖ\u200dᚱ","ᚣᚱ","ᚣᚱ","ᛠᚱ","ᛠ","ᛄᚢ",
	"ᛇ","ᛤ","ᛤ","ᚻᛚ","ᚻᚹ","ᛖ","ᚣ" ],
      [ "p","t","k","f","t̂","s","ŝ","ĉ","y","n̂",	# Diacritic
	"b","d","g","v","d̂","z","ẑ","j","w","h",
	"l","m","i","e","a","ȧ","o","ů","ă","â",
	"r","n","ē","ā","ī","u","ō","û","ǒ","ô",
	"âr","ôr","ār","r̆","ṙ","ēr","ēȧ","ū",
	"ẏ","k̂","ĝ","l̂","ŵ","ä","ö" ],
      [ "p","t","k","f","th","s","sh","ch","y","ng",	# Digraph
	"b","d","g","v","th","z","zh","j","w","h",
	"l","m","i","e","a","a","o","ou","ow","ah",
	"r","n","ee","ay","ie","u","oa","oo","oy","aw",
	"ar","or","air","ur","er","ear","ea","yoo",
	"ey","kh","gh","ll","hw","ae","oe" ],
      [ "p","t","k","f","ꜧ","ſ","ħ","tħ","i","ŋ",	# Franklin
	"b","d","g","v","ɧ","z","zħ","dħ","u","h",
	"l","m","i","e","ɑ","ɑ","oɑ","u","oɑu","ɑ̂",
	"r","n","î","ê","ɥi","ɥ","o","u","oɑi","oɑ",
	"ɑ̂r","or","êr","ɥr","er","îr","îɥ","iu",
	"i","k","g","ll","hu","ê","ɥ" ],
      [ "","","","","","","","","","",	# Quikscript
	"","","","","","","","","","",
	"","","","","","","","","","",
	"","","","","","","","","","",
	"","","","","","","","",
	"","","φ","","","","" ],
      [ "п","т","к","ф","т","с","ш","ч","й","нг",	# Cyrillic
	"б","д","г","в","т","з","ж","дж","у","х",
	"л","м","и","е","а","а","о","у","ау","а",
	"р","н","и","ей","ай","а","о","у","ой","ау",
	"ар","ор","ейр","ур","ер","ир","ия","ю",
	"и","х","г","лл","гу","ей","у" ],
      [ "p","t","k","f","θ","s","ʃ","tʃ","j","ŋ",	# CUBE
	"b","d","g","v","ð","z","ʒ","dʒ","w","h",
	"l","m","ɪ","ɛ","a","ə","ɔ","ɵ","aw","ɑː",
	"r","n","ɪj","ɛj","ɑj","ʌ","əw","ʉw","oj","oː",
	"ɑːr","oːr","ɛːr","əːr","ər","ɪːr","ɪjə","jʉw",
	"ɪj","x","ɣ","ɬ","hw","ɛː","əː" ],
      [ "p","t","k","f","T","s","S","tS","j","N",	# SAMPA
	"b","d","g","v","D","z","Z","dZ","w","h",
	"l","m","I","E","{","@","Q","U","aU","A:",
	"r","n","i:","eI","aI","V","@U","u:","OI","O:",
	"A:r","O:r","e@r","3:r","@r","i@r","i@","ju:",
	"i","x","G","K","W","e@","3:" ],
    ][alphabet-1]
  punct = { }
  if alphabet == 3:
    punct = {" ":"᛫","-":"‑",",":"᛫᛫",".":"᛬","!":"᛭",
	":":"᛬᛫",";":"᛫᛬","…":"᛫᛫᛫","(":"[",")":"]","[":"(","]":")",
	"'":"",'"':"","‘":"‹","’":"›","“":"«","”":"»" }
  if alphabet == 7:
    punct = { "(":"",")":"" }
  tran = [];
  angle = squote = dquote = eatsp = False
  for char in out:
    if char >= "𐑐" and char <= "𐒆": char = letters[ord(char)-ord("𐑐")]
    if char == "<": angle = True
    if char == ">": angle = False
    if char != " ": eatsp = False
    elif eatsp: char = ""
    if char in punct and not angle:
      if char == "'":
        char = "‘’"[squote]
        squote = not squote
      if char == '"':
        char = "“”"[dquote]
        dquote = not dquote
      char = punct[char]
      eatsp = True
    tran.append(char)
  out = "".join(tran)
  if alphabet == 3:
    for bind in """ ᚠᚩ ᚠᚪ ᚠᚫ ᚠᚱ ᚦᚱ ᚩᚢ ᚩᛉ ᚪᚱ ᚪᛉ ᚫᚢ ᚫᛉ ᚱᚱ ᚷᚩ ᚷᚪ ᚷᚫ ᚷᛚ
	ᚻᚢ ᚻᚣ ᚻᚩ ᚻᚪ ᚻᚫ ᚻᚱ ᚻᚹ ᚻᛖ ᚻᛚ ᚾᚾ ᚾᛏ ᛖᚠ ᛖᚦ ᛖᚻ ᛖᛈ ᛖᛒ ᛖᛗ ᛖᛚ ᛖᛞ
	ᛗᚢ ᛗᚣ ᛗᚩ ᛗᚪ ᛗᚫ ᛗᚱ ᛗᛗ ᛞᚢ ᛞᚣ ᛞᚩ ᛞᚪ ᛞᚫ ᛞᚱ ᛞᛖ ᛞᛗ ᛞᛞ """.split():
      out = out.replace(bind,bind[0]+"\u200d"+bind[1])
    for bind in "ᚩᚢ ᚩᚱ ᚪᚱ ᚫᚢ".split():
      out = re.sub("\u200d("+bind[0]+"\u200d"+bind[1]+")","\\1",out)
  if alphabet == 6:
    out = re.sub(r'ſ\b','s',out)
  if alphabet == 8:
    for tup in [(r'\bе','э'),("йе","е"),("йа","я"),("иа","ия"),("тс","ц")]:
      out = re.sub(tup[0],tup[1],out)

def uppercase(match):
  return match.group()[1].upper()
if alphabet not in [0,7,10]:
  out = re.sub("·.",uppercase,out).replace("‧","·")
print (out, end='')
