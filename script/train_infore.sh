GD_MODEL_DIR="/content/drive/My Drive/AI/deep_speech/models"
GD_DATA_DIR="/content/drive/My Drive/data/voice"

cd ~

echo "Download data"
cp "$GD_DATA_DIR/audiobooks.zip" .
cp "$GD_DATA_DIR/transcribe.zip" .

echo "Unzip data"
unzip -P BroughtToYouByInfoRe -q audiobooks.zip 
unzip -q transcribe.zip

echo "Copy checkpoint from google drive to local"
if [ ! -e deepspeech.pytorch ]; then
    git clone https://github.com/vudaoanhtuan/deepspeech.pytorch.git
fi
mkdir deepspeech.pytorch/models
PREV_DIR=$(ls -1r "$GD_MODEL_DIR" | head -n 1)
LAST_CHECKPOINT="$GD_MODEL_DIR/$PREV_DIR/deepspeech_2.pth.tar"
cp "$LAST_CHECKPOINT" deepspeech.pytorch/models/checkpoint.pth.tar

echo "Start trainning"
cd deepspeech.pytorch
python3 train.py \
    --train-manifest /root/train_manifest.csv \
    --val-manifest /root/test_manifest.csv \
    --labels-path vn_labels.json \
    --continue-from models/checkpoint.pth.tar \
    --finetune \
    --batch-size 32 \
    --num-workers 2 \
    --cuda \
    --checkpoint \
    --epochs 2

echo "Copy checkpoint to google drive"
rm models/checkpoint.pth.tar
NEW_DIR=$(date +'%Y-%m-%d_%H')
cp -r models "$GD_MODEL_DIR/$NEW_DIR"