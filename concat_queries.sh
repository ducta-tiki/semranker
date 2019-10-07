#!/bin/bash

IN_DIR=$1
OUT_DIR=$2

for filepath in $IN_DIR/*.csv; do
    # echo $filepath
    fbasename="${filepath##*/}"
    # echo $fbasename
    fout=$OUT_DIR/$fbasename
    echo "\n" >> $fout
    cat $filepath >> $fout
done