#!/bin/sh -x
find . -type f -name '*.edges' | while read line; do

	output =$(echo $line | cut -f 2 -d '.')
    ./../node2vec/node2vec -i:$line -o:$output -d:'100'  -v
done

