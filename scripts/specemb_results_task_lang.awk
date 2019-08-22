BEGIN {
    OFS="\t"
    lang=""
    best_dev_accuracy=0
    test_accuracy_at_best=0
    best_k=0
    trial=0
}
{
    if ($1 != lang || $2 != trial) {
        if (NR != 1) {
            print lang,trial,best_k,best_dev_accuracy,test_accuracy_at_best
        }
        lang=$1
        trial=$2
        best_k=$3
        best_dev_accuracy=$4
        test_accuracy_at_best=$5
    }

    if ($4 > best_dev_accuracy) {
        best_k=$3
        best_dev_accuracy=$4
        test_accuracy_at_best=$5
    }
}
END {
    print lang,trial,best_k,best_dev_accuracy,test_accuracy_at_best
}


