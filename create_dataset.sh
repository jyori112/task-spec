make data/processed/datasets/rcv/en.{test,dev,train}.jsonl
make data/processed/datasets/rcv/en.train.words

for lang in es de da it nl pt sv fr;
do
    make data/processed/datasets/rcv/$lang.{test,dev100}.jsonl
done

make data/processed/datasets/yelp/en.{test,dev,train}.jsonl
make data/processed/datasets/yelp/en.train.words

for lang in ja fr de;
do
    make data/processed/datasets/amazon/$lang.{dev100,test}.jsonl
done

make data/processed/datasets/amazon/en.{test,dev,train}.jsonl
make data/processed/datasets/yelp/en.train.words

for lang in en es nl tr;
do
    make data/processed/datasets/absa/$lang.{dev100,test}.jsonl
done
