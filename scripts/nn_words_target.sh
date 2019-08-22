lang=$1
shift
words=$@

MODEL_DIR=data/processed/models

echo "[general]"
echo $words| tr ' ' '\n'| python embeddings.py nn data/processed/wordemb/wiki.$lang.norm.vec --report-input --k 5

for ds in rcv yelp amazon;
do
    model=$MODEL_DIR/$ds.en.boe.specemb1
    emb=$model/emb/$lang.k$(cat $model/best_k).norm.vec
    
    if [ -e $emb ];
    then
        echo "[$ds - llm]"
        echo $words| tr ' ' '\n'| python embeddings.py nn $emb --report-input --k 5
    fi

    if [ -e $MODEL_DIR/$ds.$lang.boe.specemb1/emb/$lang.norm.vec ]
    then
        echo "[$ds - mono]"
        echo $words| tr ' ' '\n'| python embeddings.py nn $MODEL_DIR/$ds.$lang.boe.specemb1/emb/$lang.norm.vec \
            --report-input --k 5
    fi
done

