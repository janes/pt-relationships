#!/bin/bash

echo "Directions"
ggrep -P 'relation:' train_data.txt | sort | uniq -c | sort -gr

echo "No Directions"
ggrep -P 'relation:' train_data.txt | cut -d'(' -f1 | sort | uniq -c | sort -gr

echo "Total relationships"
ggrep -P 'relation:' train_data.txt | wc -l
