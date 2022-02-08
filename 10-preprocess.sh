#!/bin/bash

###
### Data preprocessing for IWSLT 2022 speech-to-text baselines
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

make_dir ${WORKDIR}/MuST-C/${SRC}-${TRG}

if test ! -e ${WORKDIR}/MuST-C/${SRC}-${TRG}/.done ; then
    env PYTHONPATH=${FAIRSEQ_ROOT} \
        ${python3} ${FAIRSEQ_ROOT}/examples/speech_to_text/prep_mustc_data_v2.py --data-root ${MUSTC_ROOT} --lang ${TRG} --output ${WORKDIR}/MuST-C --task asr st --vocab-type unigram --vocab-size-asr 8192 --vocab-size-st 32768 --cmvn-type global
    if test $? -eq 0 ; then
        touch ${WORKDIR}/MuST-C/${SRC}-${TRG}/.done
    else
        error_and_die "Failed to preprocess MuST-C dataset."
    fi
fi

exit 0
