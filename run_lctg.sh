if [ -f .env ]
then
  # Source the .env file
  source .env
else
  echo ".env file not found"
  exit 1
fi

cd scripts

if [ -z "$VERSION" ]
then
  echo "ERROR: The VERSION environment variable is not set."
  exit 1
fi

if [ -z "$OPENAI_API_KEY" ]
then
  echo "ERROR: The OPENAI_API_KEY environment variable is not set."
  exit 1
fi

if [ -z "$MODEL_PATH_0" ]
then
  echo "ERROR: The MODEL_PATH_0 environment variable is not set."
  exit 1
fi

# 実行したいpythonファイルのパスをそれぞれ定義します
INFERENCE_SCRIPT="generator.py"
FORMATTING_SCRIPT="header_footer_remover.py"
CTG_SCORE_SCRIPT="eval_ctg.py"
QUALITY_SCORE_SCRIPT="eval_quality.py"
ALL_SCORE_SCRIPT="get_scores.py"

TASKS=("summary" "ad_text" "pros_and_cons")


for TASK in ${TASKS[@]}; do
    echo "Running scripts for task: $TASK..."
    # 推論を実施するpythonファイルを実行します
    echo "Running inference script..."
    python $INFERENCE_SCRIPT --task $TASK

    # 推論結果を整形するpythonファイルを実行します
    echo "Running formatting script..."
    python $FORMATTING_SCRIPT --task $TASK

    # 整形した推論結果からスコアを獲得するpythonファイルを実行します
    echo "Running score script...(CTG)"
    python $CTG_SCORE_SCRIPT --task $TASK

    echo "Running score script...(Quality)"
    python $QUALITY_SCORE_SCRIPT --task $TASK
done

python get_scores.py

echo "All scripts executed successfully."
