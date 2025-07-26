#!/bin/bash

# Đảm bảo tmux_monitor.sh được tạo trong thư mục hiện tại
TMUX_SCRIPT="$PWD/tmux_monitor.sh"

cat > "$TMUX_SCRIPT" << 'SCRIPT'
#!/bin/bash
LOCKFILE="/tmp/tmux_monitor.lock"
SESSION_NAME="0"
WORK_DIR="rl-swarm"
VENV_PATH=".venv/bin/activate"
RUN_SCRIPT="./run_rl_swarm.sh"

# Kiểm tra khóa để tránh chạy đồng thời
if [ -e "$LOCKFILE" ]; then
    echo "Another instance is running, exiting."
    exit 1
fi
touch "$LOCKFILE"
trap 'rm -f "$LOCKFILE"; exit' EXIT INT TERM

if ! /usr/bin/tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    # Dọn dẹp tiến trình cũ (chạy dưới quyền người dùng hiện tại nếu có thể)
    pkill -f "python.*rgym_exp.runner.swarm_launcher" 2>/tmp/swarm_launcher_pkill.log || true
    sleep 2
    
    cd $HOME || exit 1
    sleep 1
    /usr/bin/tmux new-session -d -s "$SESSION_NAME" /bin/bash
    sleep 1
    /usr/bin/tmux send-keys -t "$SESSION_NAME" "cd $WORK_DIR && . $VENV_PATH && $RUN_SCRIPT" Enter
fi
SCRIPT

# Phân quyền thực thi
chmod +x "$TMUX_SCRIPT"

# Kiểm tra và thêm công việc cron
CRON_JOB="*/5 * * * * $TMUX_SCRIPT"
if ! crontab -l 2>/dev/null | grep -F "$TMUX_SCRIPT" > /dev/null; then
    echo "$CRON_JOB" | crontab -
    echo "Cron job added."
else
    echo "Cron job already exists, skipping."
fi

# Khởi động lại cron
sudo systemctl restart cron
echo "Setup complete"

# Kiểm tra crontab
crontab -l | grep tmux_monitor && echo "✓ Cron job verified" || echo "✗ Cron job failed"
