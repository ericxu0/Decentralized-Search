#!/bin/sh -x
find ./data/synthetic -type f -name 'smallworld?.txt' | while read line; do
	home="."
	extension=".emb"
	output=$home$(echo $line | cut -f 2 -d '.')$extension

    ./../node2vec/node2vec -i:$line -o:$output -d:'100'  -v
done

