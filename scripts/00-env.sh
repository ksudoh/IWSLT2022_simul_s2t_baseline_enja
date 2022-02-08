###
### Environment variables and common subroutines for IWSLT 2022
###   English-to-Japanese Simultaneous speech-to-text baselines
###
#set -e
. ${scriptdir}/01-common.sh

##
## Check environment variables
##
check_env SRC TRG WORKDIR MUSTC_ROOT

##
## Parameters
##
python3=python3
sentencepiece_vocab_size=32768
sentencepiece_num_threads=8
sentencepiece_character_coverage=0.99995
sentencepiece_pad_id=3
training_min_len=1
training_max_len=200
preprocess_workers=8

##
## Common variables
##
spm_train=${FAIRSEQ_ROOT}/scripts/spm_train.py
spm_encode=${FAIRSEQ_ROOT}/scripts/spm_encode.py

langpair=${SRC}-${TRG}
mustc_data_dir=${MUSTC_ROOT}/${langpair}/data

##
## Check directories and utilities
##
check_dir ${MUSTC_ROOT} ${mustc_data_dir} ${FAIRSEQ_ROOT}
