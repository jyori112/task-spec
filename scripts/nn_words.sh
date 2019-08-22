words=$@

MODEL_DIR=data/processed/models

echo "[general]"
echo $words| tr ' ' '\n'| python embeddings.py nn data/processed/wordemb/wiki.en.norm.vec --report-input --k 5

for ds in rcv amazon yelp;
do
    model=$MODEL_DIR/$ds.en.boe.specemb1
    emb=$model/emb/en.norm.vec

    if [ -e $MODEL_DIR/$ds.en.boe.specemb1/emb/en.norm.vec ]
    then
        echo "[$ds - mono]"
        echo $words| tr ' ' '\n'| python embeddings.py nn $MODEL_DIR/$ds.en.boe.specemb1/emb/en.norm.vec \
            --report-input --k 5
    fi
done
