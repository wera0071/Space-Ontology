python train.py --thesaurus_dir /Users/apple/Downloads/taxoenrich-master/RuWordNet \
--embeddings_path /Users/apple/Downloads/taxoenrich-master/spesial/ft_extended_vocab_20_11_20.wv \
--output_path /Users/apple/Downloads/taxoenrich-master/test_1_23_03_23 \
--lang ru --pos N --processes 8 --search_by_word --allowed_rels hypernym --topk 40 --only_leafs --train_fraction 0.2

python predict.py --model_dir /Users/apple/Downloads/taxoenrich-master/test_1_23_03_23 --input_path /Users/apple/Downloads/taxoenrich-master/spesial/ru_private_nouns.tsv \
--output_path /Users/apple/Downloads/taxoenrich-master/ru_private_nouns_predict.tsv

python eval.py --predict_path /Users/apple/Downloads/taxoenrich-master/ru_private_nouns_predict.tsv --reference_path /Users/apple/Downloads/taxoenrich-master/spesialnouns_private_subgraphs.tsv