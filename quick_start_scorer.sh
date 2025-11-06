#!/bin/bash
# ========================================
# 评估器网络（Scorer Network）快速启动脚本
# ========================================

set -e  # 遇到错误立即退出

PROJECT_ROOT="/home/xinguanze/project/ex_6_scorer/DM-scorer"
cd "$PROJECT_ROOT/projects"

echo "========================================"
echo "评估器网络（Scorer Network）快速启动"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}警告: 未检测到CUDA，训练可能会很慢${NC}"
fi

# 显示菜单
echo "请选择操作："
echo "  1) 测试环境（test_scorer_network.py）"
echo "  2) 快速训练（1%数据，用于测试）"
echo "  3) 小规模训练（10%数据）"
echo "  4) 完整训练（100%数据）"
echo "  5) 从检查点恢复训练"
echo "  6) 查看训练日志"
echo "  7) 启动TensorBoard"
echo "  q) 退出"
echo ""
read -p "请输入选项 [1-7/q]: " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}=== 测试环境 ===${NC}"
        python test_scorer_network.py
        ;;
    
    2)
        echo ""
        echo -e "${GREEN}=== 快速训练（1%数据）===${NC}"
        echo "预计时间: 5-10分钟"
        read -p "使用的GPU ID [默认0]: " gpu_id
        gpu_id=${gpu_id:-0}
        
        python run_scorer_deepmill.py \
            --alias scorer_quick_test \
            --gpu $gpu_id \
            --ratios 0.01
        ;;
    
    3)
        echo ""
        echo -e "${GREEN}=== 小规模训练（10%数据）===${NC}"
        echo "预计时间: 30-60分钟"
        read -p "使用的GPU ID [默认0]: " gpu_id
        gpu_id=${gpu_id:-0}
        
        python run_scorer_deepmill.py \
            --alias scorer_small \
            --gpu $gpu_id \
            --ratios 0.1
        ;;
    
    4)
        echo ""
        echo -e "${GREEN}=== 完整训练（100%数据）===${NC}"
        echo "预计时间: 4-8小时"
        read -p "使用的GPU ID [默认0]: " gpu_id
        gpu_id=${gpu_id:-0}
        
        echo -e "${YELLOW}警告: 这将需要较长时间，建议使用 screen 或 tmux${NC}"
        read -p "确认开始训练? [y/N]: " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python run_scorer_deepmill.py \
                --alias scorer_baseline \
                --gpu $gpu_id \
                --ratios 1.0
        else
            echo "已取消"
        fi
        ;;
    
    5)
        echo ""
        echo -e "${GREEN}=== 从检查点恢复训练 ===${NC}"
        read -p "检查点路径: " ckpt_path
        read -p "使用的GPU ID [默认0]: " gpu_id
        gpu_id=${gpu_id:-0}
        
        if [ -f "$ckpt_path" ]; then
            python run_scorer_deepmill.py \
                --alias scorer_resume \
                --gpu $gpu_id \
                --ckpt "$ckpt_path" \
                --ratios 1.0
        else
            echo -e "${RED}错误: 检查点文件不存在${NC}"
        fi
        ;;
    
    6)
        echo ""
        echo -e "${GREEN}=== 查看训练日志 ===${NC}"
        
        # 列出所有日志目录
        log_dirs=(logs/scorer_deepmill/*/models_models/ratio_*/log.csv)
        
        if [ ${#log_dirs[@]} -eq 0 ]; then
            echo "未找到训练日志"
            exit 0
        fi
        
        echo "找到以下训练日志:"
        for i in "${!log_dirs[@]}"; do
            echo "  $((i+1))) ${log_dirs[$i]}"
        done
        
        read -p "选择要查看的日志 [1-${#log_dirs[@]}]: " log_choice
        log_idx=$((log_choice-1))
        
        if [ $log_idx -ge 0 ] && [ $log_idx -lt ${#log_dirs[@]} ]; then
            tail -n 20 "${log_dirs[$log_idx]}"
        else
            echo "无效选择"
        fi
        ;;
    
    7)
        echo ""
        echo -e "${GREEN}=== 启动TensorBoard ===${NC}"
        read -p "端口号 [默认6006]: " port
        port=${port:-6006}
        
        echo "启动TensorBoard在端口 $port"
        echo "访问地址: http://localhost:$port"
        echo "按 Ctrl+C 停止"
        echo ""
        
        tensorboard --logdir=logs/scorer_deepmill --port=$port
        ;;
    
    q|Q)
        echo "退出"
        exit 0
        ;;
    
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}操作完成！${NC}"

