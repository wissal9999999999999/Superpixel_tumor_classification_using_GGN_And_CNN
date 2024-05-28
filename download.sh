#!/bin/bash

pip install gdown

zip_file_id="1BzLg4Mk5zXtIAMHkHTp9P5rf3QpinIMT"

gdown --id $zip_file_id

mkdir -p liver_data && unzip -o liver_data.zip -d liver_data


rm liver_data.zip


