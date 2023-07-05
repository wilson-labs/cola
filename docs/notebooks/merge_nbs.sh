#!/bin/bash
#set -x
dir_path=$(dirname $(realpath $0))
PREAMBLE=$dir_path/_colab_preamble.ipynb
NOTEBOOKS=$dir_path/[0-9]*.ipynb
NOTEBOOKSALL=$dir_path/[^_]*.ipynb
mkdir -p $dir_path/colabs
# cp -r $dir_path/imgs  $dir_path/colabs/imgs
for nb in $NOTEBOOKSALL
do
    nbmerge $PREAMBLE $nb -o $dir_path/colabs/$(basename $nb)
done
nbmerge $PREAMBLE $NOTEBOOKS -o $dir_path/colabs/all.ipynb