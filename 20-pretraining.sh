#!/bin/bash

###
### Data preparation script for IWSLT 2022 speech-to-text baselines
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

datadir=${WORKDIR}/MuST-C/${SRC}-${TRG}
outdir=${WORKDIR}/ASR/${SRC}-${TRG}

if test -e ${datadir}/.done ; then
    make_dir ${outdir}
    env PYTHONPATH=${FAIRSEQ_ROOT} \
        ${python3} ${FAIRSEQ_ROOT}/train.py ${datadir} \
            --tensorboard-logdir ${outdir}/logs \
            --config-yaml config_asr.yaml \
            --train-subset train_asr \
            --valid-subset dev_asr \
            --save-dir ${outdir} \
            --num-workers 4 \
            --max-tokens 40000 \
            --max-update 100000 \
            --task speech_to_text \
            --criterion label_smoothed_cross_entropy \
            --report-accuracy \
            --arch convtransformer_espnet \
            --optimizer adam \
            --lr 0.0002 \
            --lr-scheduler inverse_sqrt \
            --warmup-updates 10000 \
            --clip-norm 10.0 \
            --seed 1 \
            --update-freq 8
    if test $? -eq 0 ; then
        touch ${outdir}/.done
    else
        error_and_die "Failed to pretrain ASR."
    fi
else
    error_and_die "Data preprocessing has not been finished in ${datadir}."
fi

exit 0
