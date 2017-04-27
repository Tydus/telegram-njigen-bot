#!/bin/bash

mkdir -p retrain/2 retrain/3

cp debug/right/2/*.jpg retrain/2
cp debug/right/3/*.jpg retrain/3

# Swap 2 vs 3 in the wrong directory
cp debug/wrong/3/*.jpg retrain/2
cp debug/wrong/2/*.jpg retrain/3

