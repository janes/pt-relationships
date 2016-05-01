#!/bin/bash

echo "Directions"
grep -P 'relation:' train_data.txt | sort | uniq -c | sort -gr

echo "No Directions"
grep -P 'relation:' train_data.txt | cut -d'(' -f1 | sort | uniq -c | sort -gr

echo "Total relationships"
grep -P 'relation:' train_data.txt | wc -l
