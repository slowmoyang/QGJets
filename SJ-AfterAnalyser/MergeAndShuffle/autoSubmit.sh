#!/bin/sh
./submit_job --batch_name="MS_DJ_TRAINING" -a "dijet" "training"
./submit_job --batch_name="MS_DJ_VALIDATION" -a "dijet" "validation"
./submit_job --batch_name="MS_DJ_TEST" -a "dijet" "test"
./submit_job --batch_name="MS_ZJ_TRAINING" -a "zjet" "training"
./submit_job --batch_name="MS_ZJ_VALIDATION" -a "zjet" "validation"
./submit_job --batch_name="MS_ZJ_TEST" -a "zjet" "test"
