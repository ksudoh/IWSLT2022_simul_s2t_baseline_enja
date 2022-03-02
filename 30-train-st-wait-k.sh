#!/bin/bash

###
### ST training script for IWSLT 2022 speech-to-text baselines
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

datadir=${WORKDIR}/MuST-C/${SRC}-${TRG}
asrdir=${WORKDIR}/ASR/${SRC}-${TRG}

if test -z "${K}" ; then
    K=20
fi
outdir=${WORKDIR}/ST/${SRC}-${TRG}/wait-${K}

if test -e ${asrdir}/.done ; then
    make_dir ${outdir}
    env PYTHONPATH=${FAIRSEQ_ROOT} \
        ${python3} ${FAIRSEQ_ROOT}/train.py ${datadir} \
            --tensorboard-logdir ${outdir}/logs \
            --load-pretrained-encoder-from ${asrdir}/checkpoint_best.pt \
            --config-yaml config_st.yaml \
            --train-subset train_st \
            --valid-subset dev_st \
            --save-dir ${outdir} \
            --num-workers 8 \
            --warmup-updates 4000 \
            --max-tokens 20000 \
            --max-update 100000 \
            --task speech_to_text \
            --criterion label_smoothed_cross_entropy \
            --report-accuracy \
            --arch convtransformer_simul_trans_espnet \
            --simul-type waitk_fixed_pre_decision \
            --waitk-lagging ${K} \
            --fixed-pre-decision-ratio 7 \
            --optimizer adam \
            --lr 0.0001 \
            --lr-scheduler inverse_sqrt \
            --clip-norm 10.0 \
            --seed 2 \
            --update-freq 16
    if test $? -eq 0 ; then
        touch ${outdir}/.done
    else
        error_and_die "Failed to train ST wait-${K}."
    fi
else
    error_and_die "ASR pretraining has not been finished in ${asrdir}."
fi

exit 0
