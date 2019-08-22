dataset=$1
shift
method=$1
shift
langs=$@

for lang in $@;
    for trial in {1..3};
    do
        echo -n "$lang\t$trial\t"
        cat data/processed/models/$dataset.en-$lang.boe.$method$trial/evaluation/best.$lang.json| jq "[.dev_accuracy,.accuracy]|@tsv" -r
    done

