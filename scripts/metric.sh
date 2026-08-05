#!/bin/bash

set -e

# 获取项目根目录：scripts 的上一级目录
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$PROJECT_DIR"

# =========================================================
# Evaluation paths
# =========================================================

# 测试结果地址
EVAL_SAVE_PATH="$PROJECT_DIR/result/"

# 测试数据集地址
EVAL_TEST_PATH="$PROJECT_DIR/dataset/test"
EVAL_UNSCENE_PATH="$PROJECT_DIR/dataset/unscene"
EVAL_UNOBJECT_PATH="$PROJECT_DIR/dataset/unobject"

# 目标位置信息
OBJECT_INFO_PATH="$PROJECT_DIR/meta/map_spawnarea_info.json"

echo "Project dir: $PROJECT_DIR"
echo "Running utils/metric.py ..."

python "$PROJECT_DIR/utils/metric.py" \
  --eval_save_path "$EVAL_SAVE_PATH" \
  --eval_test_path "$EVAL_TEST_PATH" \
  --eval_unscene_path "$EVAL_UNSCENE_PATH" \
  --eval_unobject_path "$EVAL_UNOBJECT_PATH" \
  --object_info_path "$OBJECT_INFO_PATH"