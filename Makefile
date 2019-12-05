AMAZON_URL = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv'

RCV_USER = ''
RCV_PASSWORD = ''

DATA=data
INPUT=$(DATA)/input
INTERM=$(DATA)/interm
OUTPUT=$(DATA)/output
RESULT=$(DATA)/result

CLDIR=$(OUTPUT)/cl
EMBDIR=$(OUTPUT)/wordemb
DATADIR=$(OUTPUT)/dataset
MODELDIR=$(OUTPUT)/models

VOCABSIZE=200000

LANGCODE_es = spanish
LANGCODE_da = danish
LANGCODE_de = german
LANGCODE_fr = french
LANGCODE_it = italian
LANGCODE_nl = dutch
LANGCODE_sv = swedish
LANGCODE_pt = portuguese

CONTRYCODE_en = US
CONTRYCODE_ja = JP
CONTRYCODE_fr = FR
CONTRYCODE_de = DE

CROSSTASK_O = "{\"model.text_field_embedder.tokens.trainable\": true}"

.SECONDARY:

start:
	echo "Try it"

# Clean created files (except for orig)
clean:
	rm -rf $(OUTPUT)
	rm -rf vecmap
	rm -rf fastText
	rm -rf wikiextractor
	rm -rf europarl-tools

##############################
#	Install Tools
##############################

vecmap:
	git clone https://github.com/artetxem/vecmap.git
	cd vecmap && git checkout 585bf74c6489419682eef9aebe7a8d15f0873b6c

mpaligner_0.97:
	wget https://osdn.net/dl/mpaligner/mpaligner_0.97.tar.gz
	tar -zxvf mpaligner_0.97.tar.gz
	cd mpaligner_0.97 && make
	rm mpaligner_0.97.tar.gz

fastText:
	git clone https://github.com/facebookresearch/fastText.git
	cd fastText && git checkout 51e6738d734286251b6ad02e4fdbbcfe5b679382 && make

wikiextractor:
	git clone https://github.com/attardi/wikiextractor.git
	cd wikiextractor && \
		git checkout 2a5e6aebc030c936c7afd0c349e6826c4d02b871 && \
		python setup.py install

europarl-tools:
	wget http://www.statmt.org/europarl/v7/tools.tgz
	tar zxvf tools.tgz
	mv sentence-align-corpus.perl tools
	mv tools europarl-tools
	rm tools.tgz

##############################
#	For embeddings
##############################

# Download fasttext pretrained embeddings
$(INPUT)/wordemb/wiki.%.vec:
	mkdir -p $$(dirname $@)
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$*.vec \
		-O $@

# Download test set of MUSE dictionaries for evaluating CLWE
$(INPUT)/MUSE/%.test.txt:
	mkdir -p $$(dirname $@)
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$*.5000-6500.txt \
		-O $@

$(INPUT)/MUSE/%.train.txt:
	mkdir -p $$(dirname $@)
	wget https://dl.fbaipublicfiles.com/arrival/dictionaries/$*.0-5000.txt \
		-O $@

$(OUTPUT)/MUSE/%.test.txt: $(INPUT)/MUSE/%.test.txt
	mkdir -p $$(dirname $@)
	cp $< $@

$(OUTPUT)/MUSE/%.train.txt $(OUTPUT)/MUSE/%.dev.txt: $(INPUT)/MUSE/%.train.txt
	mkdir -p $$(dirname $@)
	python scripts/split_muse.py --train-out $(OUTPUT)/MUSE/$*.train.txt \
		--dev-out $(OUTPUT)/MUSE/$*.dev.txt --dev-size 500 --seed 0 \
		< $<

########## Japanese word embeddings ##########

# Download wikipedia dump on which we train Japanese word embeddings
$(INPUT)/wiki.ja.dump.bz2:
	wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 \
		-O $@

# Parse Japanese wiki dump
$(INTERM)/ja/wiki.ja.dump.txt: $(INPUT)/wiki.ja.dump.bz2 wikiextractor
	mkdir -p $$(dirname $@)
	python ./wikiextractor/WikiExtractor.py ./$(INPUT)/wiki.ja.dump.bz2 -o - \
		> $@

# Tokenize Japanese wiki dump
$(INTERM)/ja/wiki.ja.dump.tokenized: $(INTERM)/ja/wiki.ja.dump.txt
	python scripts/mecab_tokenize.py < $< > $@

# Train Japanese Word embeddings
$(INTERM)/ja/wiki.ja.vec: ./$(INTERM)/ja/wiki.ja.dump.tokenized fastText
	fastText/fasttext skipgram -input $< \
		-output $(INTERM)/ja/wiki.ja -dim 300

########## Limit vocab size of word embeddings ##########
$(EMBDIR)/wiki.%.vec: $(INPUT)/wordemb/wiki.%.vec
	mkdir -p $$(dirname $@)
	head -n `expr $(VOCABSIZE) + 1` $< | sed '1s/^.*$$/$(VOCABSIZE) 300/g' > $@

$(EMBDIR)/wiki.ja.vec: $(INTERM)/ja/wiki.ja.vec
	mkdir -p $$(dirname $@)
	head -n `expr $(VOCABSIZE) + 1` $< | sed '1s/^.*$$/$(VOCABSIZE) 300/g' > $@

########## Train CLWE ##########
$(CLDIR)/en-%/en.vec: vecmap \
	$(OUTPUT)/MUSE/en-%.test.txt \
	$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.%.vec
	mkdir -p $$(dirname $@)
	python vecmap/map_embeddings.py \
		$(EMBDIR)/wiki.en.vec $(EMBDIR)/wiki.$*.vec \
		$(CLDIR)/en-$*/en.vec $(CLDIR)/en-$*/$*.vec \
		--unsupervised --log $(CLDIR)/en-$*/log.tsv \
		--validation $(OUTPUT)/MUSE/en-$*.test.txt --cuda

########## Evaluate CLWE ##########
$(CLDIR)/en-%/test_eval.txt: $(CLDIR)/en-%/en.vec
	python vecmap/eval_translation.py $(CLDIR)/en-$*/en.vec $(CLDIR)/en-$*/$*.vec \
		-d $(OUTPUT)/MUSE/en-$*.test.txt > $@

##############################
#	Parsing Dataset
##############################

########## RCV Corpus ##########

$(INPUT)/rcv/rcv%.tar.xz:
	mkdir -p $$(dirname $@)
	wget https://ir.nist.gov/reuters/rcv$*.tar.xz \
		--http-user=$(RCV_USER) --http-password=$(RCV_PASSWORD) \
		-O $@

$(INTERM)/datasets/rcv/rcv1: $(INPUT)/rcv/rcv1.tar.xz
	mkdir -p $$(dirname $@)
	tar -xf $< -C $(INTERM)/datasets/rcv
	touch $@/lock

$(INTERM)/datasets/rcv/rcv1/lock: $(INTERM)/datasets/rcv/rcv1

$(INTERM)/datasets/rcv/rcv2: $(INPUT)/rcv/rcv2.tar.xz
	mkdir -p $$(dirname $@)
	tar -xf $< -C $(INTERM)/datasets/rcv
	mv $(INTERM)/datasets/rcv/RCV2_Multilingual_Corpus $@
	touch $@/lock

$(INTERM)/datasets/rcv/rcv2/lock: $(INTERM)/datasets/rcv/rcv2

$(INTERM)/datasets/rcv/en.parsed.jsonl: $(INTERM)/datasets/rcv/rcv1/lock
	python scripts/parse_rcv.py $(INTERM)/datasets/rcv/rcv1/ > $@

define RCV_PARSE
$(INTERM)/datasets/rcv/$(1).parsed.jsonl: $(INTERM)/datasets/rcv/rcv2/lock
	python scripts/parse_rcv.py $(INTERM)/datasets/rcv/rcv2/$(LANGCODE_$(1))/ \
		> $$@

endef

$(foreach lang,es de da it nl pt sv fr,$(eval $(call RCV_PARSE,$(lang))))

########## ABSA Corpus ##########

# Parse ABSA Corpus
$(INTERM)/datasets/absa/%.parsed.jsonl: $(INPUT)/absa/%.xml
	mkdir -p $$(dirname $@)
	python scripts/parse_absa.py < $< > $@

########## yelp ##########
$(INPUT)/yelp/yelp_academic_dataset_review.json:
	echo "Please place 'yelp_academic_dataset_review.json' in '$(INPUT)/yelp'"

$(INTERM)/datasets/yelp/en.parsed.jsonl: $(INPUT)/yelp/yelp_academic_dataset_review.json
	mkdir -p $$(dirname $@)
	cat $< | python scripts/parse_yelp.py > $@

########## Amazon ##########
$(INPUT)/amazon/amazon_reviews_multilingual_%_v1_00.tsv.gz:
	mkdir -p $$(dirname $@)
	wget $(AMAZON_URL)/amazon_reviews_multilingual_$*_v1_00.tsv.gz -O $@

define AmazonParse
$(INTERM)/datasets/amazon/$(1).parsed.jsonl: $(INPUT)/amazon/amazon_reviews_multilingual_$(CONTRYCODE_$(1))_v1_00.tsv.gz
	mkdir -p $$$$(dirname $$@)
	zcat $$< | python scripts/parse_amazon.py > $$@

endef

$(foreach lang,en de fr ja,$(eval $(call AmazonParse,$(lang))))

##############################
# 	Formatting
##############################
$(INTERM)/datasets/%.tokenized.txt: $(INTERM)/datasets/%.parsed.jsonl europarl-tools
	cat $< | jq -r '.text | gsub("[\\n]"; " ")'| europarl-tools/tokenizer.perl -l $$(basename $< | cut -d'.' -f1) \
		| tr '[A-Z]' '[a-z]' > $@

$(INTERM)/datasets/%/ja.tokenized.txt: $(INTERM)/datasets/%/ja.parsed.jsonl
	cat $< | jq -r '.text | gsub("[\\n]"; " ")'| python scripts/mecab_tokenize.py \
		| tr '[A-Z]' '[a-z]' > $@

# Get labels of RCV Corpus
$(INTERM)/datasets/%.labels.txt: $(INTERM)/datasets/%.parsed.jsonl
	cat $< | jq -r '.label' > $@

# split this dataset
$(INTERM)/datasets/%.all.jsonl: $(INTERM)/datasets/%.labels.txt $(INTERM)/datasets/%.tokenized.txt
	mkdir -p $(INTERM)/datasets/amazon
	paste $^ | jq -R 'split("\t")| {"label": .[0], "text": .[1]}| @json' -r > $@

# Shuffle dataset
$(INTERM)/datasets/%.shuf.jsonl: $(INTERM)/datasets/%.all.jsonl
	python scripts/shuffle.py --seed 1 < $< > $@

$(DATADIR)/%.dev100.jsonl: $(INTERM)/datasets/%.shuf.jsonl
	mkdir -p $$(dirname $@)
	head -n 100 $< > $@

$(DATADIR)/%.test.jsonl: $(INTERM)/datasets/%.shuf.jsonl
	mkdir -p $$(dirname $@)
	tail -n +101 $< > $@

# Special case when data is big enough to create training set
define NonEnglishDataSplit
$(DATADIR)/$(1)/$(2).test.jsonl: $(INTERM)/datasets/$(1)/$(2).shuf.jsonl
	mkdir -p $$$$(dirname $$@)
	head -n $(3) $$< > $$@

$(DATADIR)/$(1)/$(2).dev.jsonl: $(INTERM)/datasets/$(1)/$(2).shuf.jsonl
	mkdir -p $$$$(dirname $$@)
	tail -n +`expr $(3) + 1` $$< | head -n $(4) > $$@

$(DATADIR)/$(1)/$(2).train.jsonl: $(INTERM)/datasets/$(1)/$(2).shuf.jsonl
	mkdir -p $$$$(dirname $$@)
	tail -n +`expr $(3) + $(4) + 1` $$< > $$@

$(DATADIR)/$(1)/$(2).dev100.jsonl: $(DATADIR)/$(1)/$(2).dev.jsonl
	mkdir -p $$$$(dirname $$@)
	head -n 100 $$< > $$@

endef

$(eval $(call NonEnglishDataSplit,amazon,de,10000,10000))
$(eval $(call NonEnglishDataSplit,amazon,fr,10000,10000))
$(eval $(call NonEnglishDataSplit,amazon,ja,10000,10000))
$(foreach lang,da de es fr it pt sv,$(eval $(call NonEnglishDataSplit,rcv,$(lang),1000,1000)))
$(eval $(call NonEnglishDataSplit,rcv,nl,1000,100))
#$(eval $(call NonEnglishDataSplit,rcv,fr,1000,1000))
#$(eval $(call NonEnglishDataSplit,rcv,es,1000,1000))
#$(eval $(call NonEnglishDataSplit,rcv,de,1000,1000))
#$(eval $(call NonEnglishDataSplit,rcv,it,1000,1000))
#$(eval $(call NonEnglishDataSplit,rcv,sv,1000,1000))


define EnglishDataSplit
$(DATADIR)/$(1)/en.test.jsonl: $(INTERM)/datasets/$(1)/en.shuf.jsonl
	mkdir -p $$$$(dirname $$@)
	head -n $(2) $$< > $$@

$(DATADIR)/$(1)/en.dev.jsonl: $(INTERM)/datasets/$(1)/en.shuf.jsonl
	mkdir -p $$$$(dirname $$@)
	tail -n +`expr $(2) + 1` $$< | head -n $(3) > $$@

$(DATADIR)/$(1)/en.train.jsonl: $(INTERM)/datasets/$(1)/en.shuf.jsonl
	mkdir -p $$$$(dirname $$@)
	tail -n +`expr $(2) + $(3) + 1` $$< > $$@
endef

$(eval $(call EnglishDataSplit,rcv,10000,10000))
$(eval $(call EnglishDataSplit,yelp,100000,100000))
$(eval $(call EnglishDataSplit,amazon,100000,100000))

##############################
#	Build vocab
##############################
$(DATADIR)/%.words: $(DATADIR)/%.jsonl
	cat $< | python build_vocab.py --word-path $@ --label-path $(DATADIR)/$*.labels

##############################
# 	Train model
##############################
# 1. dataset
# 2. lang
# 3. model
# 4. method
define TrainModel
$(MODELDIR)/$(1).$(2).$(3).$(4)$(5)/log.jsonl: \
	$(DATADIR)/$(1)/$(2).train.jsonl \
	$(DATADIR)/$(1)/$(2).train.words \
	$(DATADIR)/$(1)/$(2).dev.jsonl
	python train.py configs/$(1).$(2).$(3).$(4).json $(OUTPUT)/models/$(1).$(2).$(3).$(4)$(5) --seed $(5)

endef

$(foreach method,genemb embffnn,\
	$(foreach lang,da de es fr it nl pt sv,\
		$(foreach trial,1 2 3,\
			$(eval $(call TrainModel,rcv,en-$(lang),boe,$(method),$(trial))))))

$(foreach method,genemb embffnn,\
	$(foreach lang,de fr ja,\
		$(foreach trial,1 2 3,\
			$(eval $(call TrainModel,amazon,en-$(lang),boe,$(method),$(trial))))))

$(foreach method,genemb embffnn,\
	$(foreach lang,es nl tr,\
		$(foreach trial,1 2 3,\
			$(eval $(call TrainModel,yelp,en-$(lang),boe,$(method),$(trial))))))

$(foreach lang,en de fr ja,\
	$(foreach trial,1 2 3,\
		$(eval $(call TrainModel,amazon,$(lang),boe,specemb,$(trial)))))
$(foreach lang,en de fr,\
	$(foreach trial,1 2 3,\
		$(eval $(call TrainModel,rcv,$(lang),boe,specemb,$(trial)))))
$(foreach trial,1 2 3,\
	$(eval $(call TrainModel,yelp,en,boe,specemb,$(trial))))

# Get best checkpoint
$(MODELDIR)/%/best: $(MODELDIR)/%
	cat $(MODELDIR)/$*/log.jsonl| jq '[.epoch,.dev_accuracy]|@tsv' -r | sort -rnk2 | head -n1 | cut -f1 > $@

$(MODELDIR)/%/best.model: $(MODELDIR)/%/best
	cp $(MODELDIR)/$*/ckpt.epoch-$$(cat $<).model $@

$(MODELDIR)/%/best.vec: $(MODELDIR)/%/best.model
	python extract_emb.py $(MODELDIR)/$* > $@

define Evaluation
$(MODELDIR)/$(1)/evaluation/$(2): $(MODELDIR)/$(1)/best.model $(DATADIR)/$(3) $(DATADIR)/$(4)
	mkdir -p $$$$(dirname $$@)
	python evaluate.py $(MODELDIR)/$(1) $(DATADIR)/$(3) --dev-path $(DATADIR)/$(4) > $$@

endef

define EvaluationWithEmb
$(MODELDIR)/$(1)/evaluation/$(2): $(MODELDIR)/$(1)/best.model $(DATADIR)/$(3) $(DATADIR)/$(4) $(5)
	mkdir -p $$$$(dirname $$@)
	python evaluate.py $(MODELDIR)/$(1) $(DATADIR)/$(3) --dev-path $(DATADIR)/$(4)\
		--emb-path $(5) > $$@

endef

# Evaluate RCV (monolingual)
$(foreach method,genemb embffnn,\
	$(foreach lang,da de es fr it nl pt sv,\
		$(foreach trial,1 2 3,\
			$(eval $(call Evaluation,rcv.en-$(lang).boe.$(method)$(trial),best.en.json,rcv/en.test.jsonl,rcv/en.dev.jsonl)))))

# Evaluate RCV (cross-lingual)
$(foreach method,genemb embffnn,\
	$(foreach lang,da de es fr it nl pt sv,\
		$(foreach trial,1 2 3,\
			$(eval $(call EvaluationWithEmb,rcv.en-$(lang).boe.$(method)$(trial),best.$(lang).json,rcv/$(lang).test.jsonl,rcv/$(lang).dev100.jsonl,$(CLDIR)/en-$(lang)/$(lang).vec)))))

# Evaluate RCV (trained on target lang)
$(foreach lang,en da de es fr it nl pt sv,\
	$(foreach trial,1 2 3,\
		$(eval $(call Evaluation,rcv.$(lang).boe.specemb$(trial),best.$(lang).json,rcv/$(lang).test.jsonl,rcv/$(lang).dev.jsonl))))

# Evaluate Amazon (monolingual)
$(foreach method,genemb embffnn,\
	$(foreach lang,de fr ja,\
		$(foreach trial,1 2 3,\
			$(eval $(call Evaluation,amazon.en-$(lang).boe.$(method)$(trial),best.en.json,amazon/en.test.jsonl,amazon/en.dev.jsonl)))))

# Evaluate Amazon (cross-lingual)
$(foreach method,genemb embffnn,\
	$(foreach lang,de fr ja,\
		$(foreach trial,1 2 3,\
			$(eval $(call EvaluationWithEmb,amazon.en-$(lang).boe.$(method)$(trial),best.$(lang).json,amazon/$(lang).test.jsonl,amazon/$(lang).dev100.jsonl,$(CLDIR)/en-$(lang)/$(lang).vec)))))

# Evaluate Amazon (trained on target lang)
$(foreach lang,en de fr ja,\
	$(foreach trial,1 2 3,\
		$(eval $(call Evaluation,amazon.$(lang).boe.specemb$(trial),best.$(lang).json,amazon/$(lang).test.jsonl,amazon/$(lang).dev.jsonl))))

# Evaluate yelp (cross-lingual)
$(foreach method,genemb embffnn,\
	$(foreach lang,nl es tr,\
		$(foreach trial,1 2 3,\
			$(eval $(call Evaluation,yelp.en-$(lang).boe.$(method)$(trial),best.en.json,absa/en.test.jsonl,absa/en.dev100.jsonl)))))

# Evaluate yelp (cross-lingual)
$(foreach method,genemb embffnn,\
	$(foreach lang,nl es tr,\
		$(foreach trial,1 2 3,\
			$(eval $(call EvaluationWithEmb,yelp.en-$(lang).boe.$(method)$(trial),best.$(lang).json,absa/$(lang).test.jsonl,absa/$(lang).dev100.jsonl,$(CLDIR)/en-$(lang)/$(lang).vec)))))

$(foreach dataset,rcv amazon yelp,\
	$(foreach trial,1 2 3,\
		$(eval $(call Evaluation,$(dataset).en.boe.genemb$(trial),best.en.json,$(dataset)/en.test.jsonl,$(dataset)/en.dev.jsonl))))

$(MODELDIR)/%/emb/en.vec: $(MODELDIR)/%/best.model
	mkdir -p $$(dirname $@)
	python extract_emb.py $$(dirname $<) > $@

define LLM
$(MODELDIR)/%/emb/$(1).k$(2).vec: \
	$(MODELDIR)/%/emb/en.vec \
	$(CLDIR)/en-$(1)/en.vec
	python llm.py $(CLDIR)/en-$(1)/en.vec $(MODELDIR)/$$*/emb/en.vec --num-neighbors $(2) \
		< $(CLDIR)/en-$(1)/$(1).vec > $$@

$(MODELDIR)/%/emb/$(1).k$(2).weights.npy: \
	$(MODELDIR)/%/emb/en.vec \
	$(CLDIR)/en-$(1)/log.tsv
	python llm.py $(CLDIR)/en-$(1)/en.vec $(MODELDIR)/$$*/emb/en.vec --num-neighbors $(2) \
		--weights-output-path $$@ \
		< $(CLDIR)/en-$(1)/$(1).vec > /dev/null

endef

define LLMMonolingual
$(MODELDIR)/%/emb/en.k$(1).vec: \
	$(EMBDIR)/wiki.en.vec
	python llm.py $(EMBDIR)/wiki.en.vec $(MODELDIR)/$$*/emb/en.vec --num-neighbors $(1) --ignore-exact-words \
		< $(EMBDIR)/wiki.en.vec > $$@

endef

# Apply LLM
$(foreach lang,da de es fr it ja nl pt sv tr,\
	$(foreach k,1 2 3 4 5 6 7 8 9 10,$(eval $(call LLM,$(lang),$(k)))))

# Apply Monolingual LLM
$(foreach k, 1 2 3 4 5 6 7 8 9 10,$(eval $(call LLMMonolingual,$(k))))

# Evaluate rcv (cross-task proj.)
$(foreach lang,da de es fr it nl pt sv,\
	$(foreach k,1 2 3 4 5 6 7 8 9 10,\
		$(foreach trial, 1 2 3,\
			$(eval $(call EvaluationWithEmb,rcv.en.boe.specemb$(trial),best.$(lang).k$(k).json,rcv/$(lang).test.jsonl,rcv/$(lang).dev100.jsonl,$(MODELDIR)/rcv.en.boe.specemb$(trial)/emb/$(lang).k$(k).vec)))))

$(foreach k,1 2 3 4 5 6 7 8 9 10,\
	$(foreach trial,1 2 3,\
		$(eval $(call EvaluationWithEmb,rcv.en.boe.specemb$(trial),best.en.k$(k).json,rcv/en.test.jsonl,rcv/en.dev.jsonl,$(MODELDIR)/rcv.en.boe.specemb$(trial)/emb/en.k$(k).vec))))

# Evaluate amazon (cross-task proj.)
$(foreach lang,de fr ja,\
	$(foreach k,1 2 3 4 5 6 7 8 9 10,\
		$(foreach trial, 1 2 3,\
			$(eval $(call EvaluationWithEmb,amazon.en.boe.specemb$(trial),best.$(lang).k$(k).json,amazon/$(lang).test.jsonl,amazon/$(lang).dev100.jsonl,$(MODELDIR)/amazon.en.boe.specemb$(trial)/emb/$(lang).k$(k).vec)))))

$(foreach k,1 2 3 4 5 6 7 8 9 10,\
	$(foreach trial,1 2 3,\
		$(eval $(call EvaluationWithEmb,amazon.en.boe.specemb$(trial),best.en.k$(k).json,amazon/en.test.jsonl,amazon/en.dev.jsonl,$(MODELDIR)/amazon.en.boe.specemb$(trial)/emb/en.k$(k).vec))))

# Evaluate yelp (cross-task proj.)
$(foreach lang,nl es tr,\
	$(foreach k,1 2 3 4 5 6 7 8 9 10,\
		$(foreach trial, 1 2 3,\
			$(eval $(call EvaluationWithEmb,yelp.en.boe.specemb$(trial),best.$(lang).k$(k).json,absa/$(lang).test.jsonl,absa/$(lang).dev100.jsonl,$(MODELDIR)/yelp.en.boe.specemb$(trial)/emb/$(lang).k$(k).vec)))))

$(foreach k,1 2 3 4 5 6 7 8 9 10,\
	$(foreach trial,1 2 3,\
		$(eval $(call EvaluationWithEmb,yelp.en.boe.specemb$(trial),best.en.k$(k).json,yelp/en.test.jsonl,yelp/en.dev.jsonl,$(MODELDIR)/yelp.en.boe.specemb$(trial)/emb/en.k$(k).vec))))

$(MODELDIR)/rcv.en.boe.specemb/evaluation/best.%.best: 
	for k in 1 2 3 4 5 6 7 8 9 10; do cat $(MODELDIR)/rcv.en.boe.specemb/evaluation/best.$*.k$$k.json; done| jq '[.accuracy,.dev_accuracy]| @tsv' -r\
		| awk '{print NR,$$1,$$2}'| tr ' ' '\t'| sort -rnk3| head -n1| cut -f1 > $@

$(MODELDIR)/rcv.en.boe.specemb/evaluation/best.%.best.json: $(MODELDIR)/rcv.en.boe.specemb/evaluation/best.%.best
	cat $(MODELDIR)/rcv.en.boe.specemb/evaluation/best.$*.k$$(cat $<).json > $@

############################################################
#	Linear mapping
############################################################
define LinearMap
$(MODELDIR)/%/emb/$(1).unconst.vec: \
	$(MODELDIR)/%/emb/en.vec \
	$(CLDIR)/en-$(1)/en.vec \
	$(CLDIR)/en-$(1)/$(1).vec
	python llm.py $(CLDIR)/en-$(1)/en.vec $(MODELDIR)/$$*/emb/en.vec --linear \
		< $(CLDIR)/en-$(1)/$(1).vec > $$@

$(MODELDIR)/%/emb/$(1).orth.vec: \
	$(MODELDIR)/%/emb/en.vec \
	$(CLDIR)/en-$(1)/log.tsv 
	python llm.py $(CLDIR)/en-$(1)/en.vec $(MODELDIR)/$$*/emb/en.vec --linear --orthogonal \
		< $(CLDIR)/en-$(1)/$(1).vec > $$@

endef

$(foreach lang,da de es fr it ja nl pt sv tr,\
	$(eval $(call LinearMap,$(lang))))

# Evaluate rcv (cross-task proj.)
$(foreach lang,da de es fr it nl pt sv,\
	$(foreach trial, 1 2 3,\
		$(eval $(call EvaluationWithEmb,rcv.en.boe.specemb$(trial),$(lang).unconst.json,rcv/$(lang).test.jsonl,rcv/$(lang).dev100.jsonl,$(MODELDIR)/rcv.en.boe.specemb$(trial)/emb/$(lang).unconst.vec))))

$(foreach lang,da de es fr it nl pt sv,\
	$(foreach trial, 1 2 3,\
		$(eval $(call EvaluationWithEmb,rcv.en.boe.specemb$(trial),$(lang).orth.json,rcv/$(lang).test.jsonl,rcv/$(lang).dev100.jsonl,$(MODELDIR)/rcv.en.boe.specemb$(trial)/emb/$(lang).orth.vec))))

# Evaluate amazon (cross-task proj.)
$(foreach lang,de fr ja,\
	$(foreach trial, 1 2 3,\
		$(eval $(call EvaluationWithEmb,amazon.en.boe.specemb$(trial),$(lang).unconst.json,amazon/$(lang).test.jsonl,amazon/$(lang).dev100.jsonl,$(MODELDIR)/amazon.en.boe.specemb$(trial)/emb/$(lang).unconst.vec))))

$(foreach lang,de fr ja,\
	$(foreach trial, 1 2 3,\
		$(eval $(call EvaluationWithEmb,amazon.en.boe.specemb$(trial),$(lang).orth.json,amazon/$(lang).test.jsonl,amazon/$(lang).dev100.jsonl,$(MODELDIR)/amazon.en.boe.specemb$(trial)/emb/$(lang).orth.vec))))

# Evaluate yelp (cross-task proj.)
$(foreach lang,nl es tr,\
	$(foreach trial, 1 2 3,\
		$(eval $(call EvaluationWithEmb,yelp.en.boe.specemb$(trial),$(lang).unconst.json,absa/$(lang).test.jsonl,absa/$(lang).dev100.jsonl,$(MODELDIR)/yelp.en.boe.specemb$(trial)/emb/$(lang).unconst.vec))))

$(foreach lang,nl es tr,\
	$(foreach trial, 1 2 3,\
		$(eval $(call EvaluationWithEmb,yelp.en.boe.specemb$(trial),$(lang).orth.json,absa/$(lang).test.jsonl,absa/$(lang).dev100.jsonl,$(MODELDIR)/yelp.en.boe.specemb$(trial)/emb/$(lang).orth.vec))))

#$(foreach lang,da de es fr it nl pt sv,\
#	$(foreach k,1 2 3 4 5 6 7 8 9 10,\
#		$(foreach trial, 1 2 3,\
#			$(eval $(call EvaluationWithEmb,\
#					rcv.en.boe.specemb$(trial),\
#					$(lang).orth.json,\
#					rcv/$(lang).test.jsonl,\
#					rcv/$(lang).dev100.jsonl,\
#					$(MODELDIR)/rcv.en.boe.specemb$(trial)/emb/$(lang).orth.vec)))))

%.norm.vec: %.vec
	cat $*.vec| grep -E "^[[:alpha:]]+\s"| python embeddings.py format > $@

$(MODELDIR)/%/best_k:
	cat $(MODELDIR)/$*/evaluation/en.k$(wildcard "1 2 3 4 5 6 7 8 9 10").json| jq .dev_accuracy| awk 'BEGIN { OFS="\t" } {print NR,$$0}'| sort -rnk 2| head -n 1| cut -f1 > $@
