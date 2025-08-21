echo "Download Wiki"
mkdir -p wikidata && cd wikidata
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2
cd -

echo "gpt2-vocab"
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt