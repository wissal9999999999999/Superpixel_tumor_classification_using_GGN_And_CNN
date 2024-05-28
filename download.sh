#!/bin/bash

pip install gdown

zip_file_id="1ExZ-N4_RcMzNZO2uI-wo7pJqAFY135X9"

gdown --id $zip_file_id

mkdir -p liver_data && unzip -o liver_data.zip -d liver_data


rm liver_data.zip


