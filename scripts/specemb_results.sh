dataset=$1
shift
langs=$@

for lang in $@;
    for trial in {1..3};
        for k in {1..10};
        do
            echo -n "$lang\t$trial\t$k\t"
            cat data/processed/models/$dataset.en.boe.specemb$trial/evaluation/best.$lang.k$k.json| jq "[.dev_accuracy,.accuracy]|@tsv" -r
        done
