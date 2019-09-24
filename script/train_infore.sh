cd ~
cp /content/drive/My\ Drive/data/voice/audiobooks.zip .
cp /content/drive/My\ Drive/data/voice/transcribe.zip .
unzip -P BroughtToYouByInfoRe -q audiobooks.zip 
unzip -q transcribe.zip
if [ ! -e deepspeech.pytorch ]; then
    git clone https://github.com/vudaoanhtuan/deepspeech.pytorch.git
fi
mkdir deepspeech.pytorch/models
LAST_CHECKPOINT="$(ls -1r /content/drive/My\ Drive/AI/deep_speech/models/deepspeech_*.pth.tar | head -n 1)"
cp "$LAST_CHECKPOINT" deepspeech.pytorch/models
LAST_CHECKPOINT="$(ls -t deepspeech.pytorch/models | head -n 1)"
cd deepspeech.pytorch
python3 train.py \
    --train-manifest /root/train_manifest.csv \
    --val-manifest /root/test_manifest.csv \
    --labels-path vn_labels.json \
    --continue-from models/$LAST_CHECKPOINT \
    --finetune \
    --batch-size 32 \
    --num-workers 2 \
    --cuda \
    --checkpoint \
    --epochs 2
rm $LAST_CHECKPOINT
cp models/* /content/drive/My\ Drive/AI/deep_speech/models/