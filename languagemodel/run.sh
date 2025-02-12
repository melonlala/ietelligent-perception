!bin/bash
# source /lustre/home/acct-stu/stu294/workspace/languagemodel/bin/activate
python main.py --data data/gigaspeech --cuda  --model RNN_RELU --epochs 20 --lr 5
python main.py --data data/gigaspeech --cuda  --model GPT --epochs 20