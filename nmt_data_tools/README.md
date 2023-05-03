# jjyaoao 的文本工具箱

> 特别鸣谢: https://github.com/MiuGod0126

下载并安装依赖:

```bash
git clone git@github.com:jjyaoao/jjyaoao_tools.git
cd jjyaoao_tools/nmt_data_tools
pip install -r requirements.txt
```

### 1.分词

#### 中文处理：

> 英文也可以用

```bash
若要在linux环境测试，请使用
vim nmt_tools/cut_zh.sh
### Esc进入命令行运行模式
### : set ff=unix
```

```bash
# 1.单进程
python nmt_tools/cut_zh.py src_file tgt_file

# 2.多进程
workers=4
bash nmt_tools/cut_zh.sh  $workers src_file tgt_file

# 3.example:
bash nmt_tools/cut_zh.sh 4 data/train.zh data/train.tok.zh
```

```bash
# result:
故事的小黄花
从出生那年就飘着
=========》
故事 的 小黄花
从 出生 那年 就 飘 着
```

---

#### 泰文处理：

> 格式完全一样，只是从 zh 改为 th

```bash
bash nmt_tools/cut_th.sh 4 data/train.zh data/train.tok.zh
```

```bash
# result:
ดอกไม้สีเหลืองเล็ก ๆ น้อย ๆ ของเรื่อง
ลอยตัวจากปีเกิด
=========》
ดอกไม้ สีเหลือง เล็ก ๆ น้อย ๆ ของ เรื่อง
ลอยตัว จาก ปีเกิด
```

### 2.词表转换

json->vocab(paddle)->dict(fairseq)

**1. json2dict，不再需要从 vocab 中转，且能保留词频信息。**

**2.json2vocab 和 json2dict 加入 min_freq，参数，支持按照频率过滤词表。**

```bash
# 1.json 转 paddle vocab
#python nmt_tools/json2vocab.py $infile $outfile $min_freq(optional)
python nmt_tools/json2vocab.py data/train.bpe.zh.json data/vocab.zh


# 2.json 转 fairseq dict
#python nmt_tools/json2dict.py $infile $outfile $min_freq(optional)
python nmt_tools/json2dict.py data/train.bpe.zh.json data/dict1.zh.txt

# 3.paddle vocab 转 fairseq dict
# python nmt_tools/vocab2dict.py $infile $outfile $min_freq(optional)
python nmt_tools/vocab2dict.py data/vocab.zh data/dict.zh.txt

# 4.fairseq dict 转  paddle vocab
# python nmt_tools/dict2vocab.py $infile $outfile
python nmt_tools/dict2vocab.py data/dict.zh.txt data/vocab.zh
```

```bash
# 数据类型一览:
train.bpe.zh.json
"<EOS>": 0,
"<GO>": 1,

vocab.zh
<GO>
<UNK>
的

dict.zh.txt
。 100000
， 99999
&quot; 99998

dict1.zh.txt
, 4
。 5
， 6
```

### 3.过滤

1.**⭐ 语言标识过滤**

使用 fasttext

```bash
# 1.下载权重（放nmt_data_tools目录下）
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# 2.过滤平行语料
python ./nmt_tools/data_filter.py --src-lang $SRC --tgt-lang $TRG --in-prefix data/train --out-prefix data/trainlang --threshold 0.5

# 3.example
python ./nmt_tools/data_filter.py --src-lang zh --tgt-lang en --in-prefix data/train --out-prefix data/trainlang --threshold 0.1
# result: lang id filter| [967/1000] samples retained, [33/1000] were deleted.
```

```bash
# 有时句子里会掺入其他语言的词汇（比如：“Today天气real不错”），语言标识过滤用fasttext识别src和tgt的语言id，若任一边不符合原来的id，则删除这对语料。

# fasttext使用案例
import fasttext
model_path="lid.176.bin"
model=fasttext.load_model(model_path)
labels,scores=model.predict(["今天天气真不错","Today天气real不错"])
print(labels,scores)

[['__label__zh'], ['__label__en']] [array([0.99306196], dtype=float32), array([0.32698047], dtype=float32)]
# 可以看到当句子较短时，仅掺两个英文词汇便被认为是英文__label__en，这样"Today天气real不错"所在的语料对会被过滤掉。除了看__label__zh是否正确，还可以考虑置信度高低，如label正确，但置信度0.4<threshod=0.5也可以过滤掉（可以设低点）

```

**2.长度过滤**

替代 moses 的 clean-corpus，打印统计信息。

```bash
# 长度检测 (用于tokenize后，bpe前)
# python nmt_tools/check_pair.py <in_prefix> <SRC> <TRG>  <upper> <ratio> <write_trash>(0/1)
python nmt_tools/check_pair.py data/train.tok zh th  175 1.5 0
# 默认write_trash=0，只打印，不写trash；0时会把范围外的异常数据写入 <inprefix.trash.lang>

# 长度过滤 (替代moses的clean-corpus；一般用于bpe之前)
# 有些明显长度不匹配的能过滤掉，比如一个3个词，另一边30个
#（1-250，比例1:2.5或2.5:1）,长度过滤往往丢的数据最多，多手下留情哈。
python nmt_tools/length_filter.py --src-lang zh --tgt-lang th --in-prefix  data/train.bpe --out-prefix data/train.clean --low 1 --up 200 --ratio 1.5 --remove-bpe --wt
# --remove-bpe可选，开启后判断长度时删掉@@ ，不影响写入； --wt可选，效果同check_pair的write_trash
```

```bash
check 结果：
              ratio       src_len       tgt_len
count  1.503344e+06  1.503344e+06  1.503344e+06
mean   1.613559e+00  7.379560e+00  8.033880e+00
std    9.832292e-01  5.450975e+00  5.950167e+00
min    1.000000e+00  1.000000e+00  1.000000e+00
25%    1.142857e+00  4.000000e+00  5.000000e+00
50%    1.333333e+00  6.000000e+00  7.000000e+00
75%    1.714286e+00  9.000000e+00  1.000000e+01
max    8.500000e+01  5.100000e+02  5.010000e+02
84 lines len > 175, 65409 lines ratio > 3.0.
step info: {'1.5': 494654, '2': 194131, '2.5': 118475, '3': 65409}
```

### 4.批量提取 xml 和 sgm

#### 4.1 提取单个 xml：

```bash
# 1.command:
python nmt_tools/process_xml.py $infile $outfolder

# 2.example:
python nmt_tools/process_xml.py data/xml/bgzh/val.bg-zh.bg.xml data/
# result:
#total 1000 lines.
#write to data/val.bg-zh.bg.txt success.
```

```bash
<seg id="1">Можем да приключваме.</seg>
<seg id="2">Няма да ни липсваш.</seg>
<seg id="3">Свикнал е да изнася лекции.</seg>
=====》
Можем да приключваме.
Няма да ни липсваш.
Свикнал е да изнася лекции.
```

#### 4.2 ⭐ 提取含 xml 的文件夹

```bash
# 1.command:
bash nmt_tools/process_xml_folder.sh <infolder> <outfolder>

# 2.example:
bash nmt_tools/process_xml_folder.sh data/xml/ xml_out
```

```bash
############### input ############### ：
data/xml
├── bgzh
│   ├── val.bg-zh.bg.xml
│   └── val.bg-zh.zh.xml
└── ruzh
    ├── val.ru-zh.ru.xml
    └── val.ru-zh.zh.xm
############### output ############### ：
xml_out/
├── bgzh
│   ├── val.bg-zh.bg.txt
│   └── val.bg-zh.zh.txt
└── ruzh
    ├── val.ru-zh.ru.txt
    └── val.ru-zh.zh.txt
```

### 5.⭐ 写入 xml：

```bash
# 1.生成xml
python nmt_tools/write_xml.py data/train.en data/result.xml
```

```bash
# input:  So this is our last stop, and we came back to our headquarter in Beijing.
# output:
<tstset setid="nestest2019" srclang="zh" trglang="en">
  <DOC docid="news" sysid="1">
    <p>
      <seg id="1">So this is our last stop, and we came back to our headquarter in Beijing.</seg

# 2.再在xml开头添加
<?xml version="1.0" encoding="UTF-8"?>
# 即是标准的xml文件
```

### 6.流式输入处理

```bash
# 1.command:
python nmt_tools/stream_preprocess.py <infile> <outfile> <1(towhole)|2(tostream)> <lang>

# 2.example:
# 流式->整句（data/whole.zh）
python nmt_tools/stream_preprocess.py data/stream.zh data/whole.zh 1 zh
# 整句->流式（data/stream2.zh）
python nmt_tools/stream_preprocess.py data/whole.zh data/stream2.zh 2 zh
```

```bash
如：
大
大家
大家晚         <=========> 	大家晚上好
大家晚上
大家晚上好
```

### 7.合并、拆分语料

#### 7.1 合并

```bash
# 1.python
# python {sys.argv[0]} <infile1> <infile2> <outfile> <sep> (space/tab)
# 空格分隔
python nmt_tools/merge.py data/train.zh data/train.en data/outfile.txt space
# 制表符分割
python nmt_tools/merge.py data/train.zh data/train.en data/outfile.txt tab

# 2.shell
paste data/train.zh data/train.en > data/outfile.txt
# paste -d ' ' 空格分割不好用，paste -d默认参数是\t
```

#### 7.2 拆分

```bash
# src
cut -f 1 data/outfile.txt > data/cut.zh
# tgt
cut -f 2 data/outfile.txt > data/cut.en
```

> 合并拆分均是指交叉语料，例如中英交叉合并，中英交叉拆分

### 8.数据划分

划分训练和验证集

```bash
# 1.command:
python nmt_tools/train_dev_split.py <src-lang> <tgt-lang> <inprefix> <outfolder> <dev len>

# 2.example:
python nmt_tools/train_dev_split.py zh en data/train data 500
# 从train中随机取500条到data/dev.zh/en，其余的data/train.zh/en,result:
# dev为验证集
write to data\train.zh success.
write to data\train.en success.
write to data\dev.zh success.
write to data\dev.en success.
```

### 9.上、下采样

#### 9.1 上采样

```bash
# 1.command:
python nmt_tools/upsample.py <src-lang> <tgt-lang> <inprefix> <outfolder> <upsample len>

# 2.example:
# train为文件名，zh、en为文件后缀
# 将不够的语句不断随机刷新凑够10000
python nmt_tools/upsample.py zh en data/train data 10000
# result:
write to data\upsample.zh success.
write to data\upsample.en success.
```

#### 9.2 下采样

直接使用上文**8.数据划分**，取 dev.lang 作为下采样结果

### 10.打乱平行语料

```bash
# 1.command:
python nmt_tools/shuffle_pair.py <src_lang> <tgt_lang> <data_prefix> <out_folder>

# 2.example:
python nmt_tools/shuffle_pair.py zh en data/train data/
# result:
write to data/shuffle.zh success.
write to data/shuffle.en success.
```

### 11.去重

双语去重。（无序，比有序快 25%）

```bash
# python nmt_tools/deduplicate_pairs.py  <in_prefix> <src_lang> <tgt_lang>  <workers>
# 会默认写到in_prefix.dedup.lang里,example:
python nmt_tools/deduplicate_pairs.py  data/train zh en  4
# 文件名train.dedip.zh/en
```

单语多文件去重 。（无序）

```bash
# 1.command:
python nmt_tools/deduplicate_lines.py --workers $workers files > $outfile

# 2.example:
python nmt_tools/deduplicate_lines.py --workers 4 data/upsample.zh > data/dedup.zh
wc data/dedup.zh
# 行数为493、单词数499、字节数47809  data/dedup.zh
```

### 12.⭐ 翻译数据处理

以 zh en 为例进行数据处理：

(未完成，后续魔改)

```bash
bash preprocess.sh
```

### 13.后处理

对英文 detokenize+detruecase

```bash
# bash postprocess.sh <prefix>
bash postprocess.sh data/zhen_bpe/train.bpe
```

### 14.fast_align 抽取词典

```bash
# 分词
bash nmt_tools/cut_zh.sh 4 data/train.zh data/train.tok.zh
bash nmt_tools/cut_zh.sh 4 data/dev.zh data/dev.tok.zh
mv data/train.tok.zh data/train.zh
mv data/dev.tok.zh data/dev.zh
# 抽取词典 （data/fast_align/dict.zh-en）
bash fast_align_dict.sh
# 查看结果
head -n 10 data/fast_align/dict.zh-en
我们    we      148
的      of      142
的      the     120
这个    the     100
是      is      100
，      and     94
一个    a       84
的      in      82
我们    our     76
可以    can     74
```

目前已经可以抽取词典，后续准备添加停用词过滤，然后清洗词典，或者检测并保留些实体词，就不放在抽取词典上了。

### 15.RAS

随机对齐替换。用词典将 source 的词典词按一定几率替换为其他语言的同义词。

```bash
python nmt_tools/replace_word_bilingual.py --langs de;en --dict-path dic --data-path data  --prefix train --num-repeat 1 --moses-detok --replace-prob 0.3 --vocab-size 1000

tree:
/dic
	de-en.txt # de_word en_word \n (空格分隔)
/data
	train.src # <lang_id> src_text
	train.tgt  # <lang_id> tgt_text
```

**完整文档和 demo 参考[ras_sample](https://github.com/MiuGod0126/nmt_data_tools/blob/main/ras_sample/README.md)**

### 16.⭐ 罕见词、乱码过滤（中文）

1.对于部分乱码，编码成 gbk 后会报错，如此可以剔除一部分：

```bash
def check_is_encode_error(string):
    try:
        string.encode('gbk')
    except UnicodeEncodeError:
        return True
    return False
# eg：s="硂或ㄓ и籔羆琌畉˙ぇ换"
# print(check_is_encode_error(s)) # True
```

2.对于不会报错的罕见词，可以在文本进行分词后用 unicode 编码区分中文和其他字符，大多数情况是符号或者日语，所以用日语的 unicode 来区分是不是正常的词。

```bash
def is_all_japanese(strs):
    for _char in strs:
        if not '\u0800' <= _char <= '\u4e00':
            return False
    return True

# 如得到下面奇怪的词表：
'''
['ㄛ', '①', '⼈', '┅', '◎', '●', 'ㄧ', '〈', '〖', '≥', '②', '\u200b', '•', '⼀', 'ㄓ', '〇', 'ㄚ', '③', '⻔', '⼝', 'Ⅲ', '④', '⑤', '│', '∩', 'ぃ', '⼦', 'ส', '์', 'ร', 'ぎ', '㖊', 'ㄩ', '≤', '⾕', 'ぐ', '⊿', 'Ⅳ', 'ย', '⾯', 'ヶ', '∣', 'ㄤ', '⊙', '▲', 'ㄍ', 'ㄞ', '⾃', '⾮', '※', 'จ', 'す', 'れ', '✘', 'レ', '⼩', 'ㄈ', 'ิ', '⽤', '⑦', '⑥', '❤', 'ㄇ', 'ㄐ', '⽽', 'ㄘ', 'ㄗ', '⼴', '⼊', '‰', '⼼', '⼯', '▓', '┕', '⾛', '〜', '䴕', 'Ⅵ', '⽣', 'い', 'น', 'ゅ', 'え', 'ㄅ', 'ㄎ', '∕', '㎡', 'ㄥ', 'わ', 'Ⅴ', '✔', 'ไ', 'พ', 'ค', 'ต', 'ม', 'า', 'อ', 'ซ', 'เ', '㖞', '⽿', '⽶', '⻄', '⽬', '⽂', 'ฮ', 'ว', 'ง', 'ุ', '้', 'が', 'ば', 'ヴ', 'か', '╩', '㓥', '㎝', '═', 'オ', '㈱', '┣', '⑩', 'ㄖ', 'ㄌ', 'ㄆ', 'ボ', 'ヘ', 'も', 'デ', 'ㄨ', 'ㄉ', 'ㄠ', 'ァ', 'む', 'ェ', 'ゴ', '√', '⽞', '⼜', '⾜', 'Ⅷ', 'Ⅶ', '∨', '≠', '⻅', '⾄', '∪', 'Ⅸ', 'ะ', 'ื', 'ั', 'ท', 'ี', '่', '⑧', 'ㄙ', '⼤', 'ト', 'ッ', 'プ', 'グ', 'イ', 'ン', 'を', '⾸', '⺟', '⾏', '・', '⽉', '⽴', 'ㄒ', 'ㄢ', 'の', 'ど', 'こ', 'ん', 'に', 'ち', '╰', '╭', '″', 'せ']
'''
```

找出含这些词的句子，这部分句子有很大概率是无意义的句子（全是乱码），也有可能只是包含些特殊符号，我们对这些嫌疑句进行打分，规则如下：

si=(w1,w2,...,wn)

$$ Score(i)=\frac{1}{n}\sum*{k=0}^{k=n}I(freq*{k}<100)\* \frac{（100-freq\_{k})}{100} $$ 对于第 i 个句子 si，有 n 个 word。句子分数 Score(i)为，n 个 word 的分数求平均。指示函数 I 当词频小于 100 时为 1， 对于词频 freq>100 的词，分数为 0；对于 freq<100 的词，分数为归一化的 100-freq，这样词频低就分数高；若乱码多，基本上分数很容易超过 0.5,俺把超过 0.45 的都当初乱码丢掉了。对于 score<0.45 的嫌疑句，俺按规则删掉些乱码词后放回原语料。

```bash
# python nmt_tools/zh_abnormal_filter.py --zh-lang zh --other-lang th --in-prefix data_zhth/train.tok --out-prefix data_zhth/train.clean --threshold 0.45 --min-freq 20 --wt
# 用在tokenize之后，bpe前
# --wt可选，加了后会把嫌疑句存到out_prefix.update.zh 和out_prefix.trash.zh
'''
update: (score<0.45)

Score: [0.140], Sentence: [◎ 我 都 会 等 你
]Score: [0.213], Sentence: [◎ 在 那些 我 无法 忘怀 的
]Score: [0.210], Sentence: [◎ 所有 熟悉 的
]Score: [0.280], Sentence: [◎ 老 地方
]Score: [0.210], Sentence: [◎ 等 着 你

trash: (score>0.45)

Score: [0.829], Sentence: [※ 涴 汊 唒 ＃ § 庈 部 塑 毽 斯 剟 覂 ※ 扥 酗 盛 赽 わ 晒 赵塑 §
]Score: [0.475], Sentence: [395 ) 若 吾 起舞 吾 が 舞 え ば
]Score: [0.790], Sentence: [惕 К - 凌 岆 勤祥 れ
]Score: [0.784], Sentence: [膻寿 炵 ㄛ 梗 温 婓 陑 奻
]Score: [0.741], Sentence: [辣 茩 蝠 霜 ﹛ QQ ㄩ 31946467
]Score: [0.823], Sentence: [祥 岆 珨 跺 艘 善 か 谣艺 羹 憩霜谙 厄腔 伎 寤 橾 啄
'''
```
