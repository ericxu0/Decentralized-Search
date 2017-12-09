#!/bin/sh -x
find . -type f -name '*.edges' | while read line; do
	home="."
	extension=".emb"
	output=$home$(echo $line | cut -f 2 -d '.')$extension

    ./../node2vec/node2vec -i:$line -o:$output -d:'256'  -v
done

