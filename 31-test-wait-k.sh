#!/bin/bash

###
### Train a wait-k model
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
SIMULEVAL_ROOT=${basedir}/SimulEval
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

if test -z "${K}" ; then
    K=20
fi
model_dir=${WORKDIR}/ST/${SRC}-${TRG}/wait-${K}

decoder_options=""
if test ${TRG} = "ja" ; then
    decoder_options="${decoder_options} --sacrebleu-tokenizer ja-mecab"
    decoder_options="${decoder_options} --no-space"
fi
if test -z "${SIMULEVAL_PORT}" ; then
    SIMULEVAL_PORT=12321
fi

if test ! -s ${model_dir}/checkpoint_best.pt ; then
    error_and_die "No model checkpoints are found in ${model_dir}."
elif test ! -e ${model_dir}/.done ; then
    warn "The model training has not completed. The best intermediate checkpoint will be used."
fi

for task in tst-COMMON tst-HE ; do
    if test -e ${model_dir}/${task}/.done ; then
        notice "results for ${task} is found in ${model_dir}/${task}."
    else
        env PYTHONPATH="${FAIRSEQ_ROOT}:${SIMULEVAL_ROOT}" \
            ${python3} ${SIMULEVAL_ROOT}/bin/simuleval \
                --agent ${FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py \
                --source ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}/${task}.wav_list \
                --target ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}/${task}.${TRG} \
                --data-bin ${WORKDIR}/MuST-C/${SRC}-${TRG} \
                --config config_st.yaml \
                --model-path ${model_dir}/checkpoint_best.pt \
                --output ${model_dir}/${task} \
                --scores ${decoder_options} \
                --port ${SIMULEVAL_PORT} \
                --gpu
        if test $? -eq 0 ; then
            touch ${model_dir}/${task}/.done
        else
            warn "test for ${task} failed in ${model_dir}."
        fi
    fi
done
