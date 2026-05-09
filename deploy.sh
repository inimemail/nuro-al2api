#!/usr/bin/env bash

set -Eeuo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

APP_NAME="AIClient2API"
DEFAULT_INSTALL_PATH="/opt/aiclient2api"
ENV_RECORD_FILE="/etc/aiclient2api_env"

CRON_TAG_BEGIN="# AICLIENT2API_BACKUP_BEGIN"
CRON_TAG_END="# AICLIENT2API_BACKUP_END"
BACKUP_LOG="/var/log/aiclient2api_backup.log"

CONTAINER_NAME="aiclient2api"
SERVICE_NAME="aiclient-api"
IMAGE_NAME="aiclient2api-local:latest"
SOURCE_DIR_NAME="source"
SOURCE_REPO_URL="${SOURCE_REPO_URL:-https://github.com/inimemail/nuro-al2api.git}"
SOURCE_REPO_BRANCH="${SOURCE_REPO_BRANCH:-main}"

DEFAULT_WEB_PORT="3000"
GEMINI_PORT_RANGE="8085-8087"
CODEX_PORT="1455"
KIRO_PORT_RANGE="19876-19880"

ADMIN_PASS=""

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
warn() { echo -e "\033[33m[WARN]\033[0m $1" >&2; }
err()  { echo -e "\033[31m[ERROR]\033[0m $1" >&2; }
die()  { echo -e "\033[31m[FATAL]\033[0m $1" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "系统缺少核心依赖: $1"
}

get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1"
}

valid_port() {
    local p="$1"
    [[ "$p" =~ ^[0-9]+$ ]] && [[ "$p" -ge 1 ]] && [[ "$p" -le 65535 ]]
}

port_in_use() {
    local p="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${p}$"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${p}$"
    else
        return 1
    fi
}

find_free_port() {
    local start="$1"
    local p="$start"
    while [[ "$p" -le 65535 ]]; do
        if ! port_in_use "$p"; then
            echo "$p"
            return 0
        fi
        p=$((p + 1))
    done
    return 1
}

find_free_port_range() {
    local start="$1"
    local count="$2"
    local p="$start"
    local ok i
    while [[ $((p + count - 1)) -le 65535 ]]; do
        ok=1
        for ((i=0; i<count; i++)); do
            if port_in_use $((p + i)); then
                ok=0
                break
            fi
        done
        if [[ "$ok" -eq 1 ]]; then
            echo "${p}-$((p + count - 1))"
            return 0
        fi
        p=$((p + 1))
    done
    return 1
}

resolve_oauth_ports() {
    local gemini_start codex_start kiro_start
    gemini_start="${GEMINI_PORT_RANGE%%-*}"
    codex_start="$CODEX_PORT"
    kiro_start="${KIRO_PORT_RANGE%%-*}"

    GEMINI_PORT_RANGE="$(find_free_port_range "$gemini_start" 3)" || die "No free 3-port range found for Gemini/Antigravity OAuth"
    CODEX_PORT="$(find_free_port "$codex_start")" || die "No free port found for Codex OAuth"
    KIRO_PORT_RANGE="$(find_free_port_range "$kiro_start" 5)" || die "No free 5-port range found for Kiro OAuth"
}

docker_compose_cmd() {
    if command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    else
        die "未探测到 Docker Compose。请先安装 docker compose 或 docker-compose。"
    fi
}

get_workdir() {
    [[ -f "$ENV_RECORD_FILE" ]] && cat "$ENV_RECORD_FILE" || true
}

get_script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd
}

find_project_root() {
    if [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/Dockerfile" && -f "${PROJECT_ROOT}/package.json" ]]; then
        cd "$PROJECT_ROOT" >/dev/null 2>&1 && pwd
        return 0
    fi

    local script_dir
    script_dir="$(get_script_dir)"
    if [[ -f "${script_dir}/Dockerfile" && -f "${script_dir}/package.json" ]]; then
        echo "$script_dir"
        return 0
    fi

    if [[ -f "${PWD}/Dockerfile" && -f "${PWD}/package.json" ]]; then
        pwd
        return 0
    fi

    return 1
}

sync_project_source() {
    local workdir="$1"
    local project_root=""
    local dest="${workdir}/${SOURCE_DIR_NAME}"

    if project_root="$(find_project_root)"; then
        mkdir -p "$dest"

        info "Syncing current source to ${dest} ..."
        tar \
            --exclude='./.git' \
            --exclude='./node_modules' \
            --exclude='./logs' \
            --exclude='./backups' \
            --exclude='./configs' \
            --exclude='./.env' \
            --exclude='./.env.*' \
            -cf - -C "$project_root" . | tar -xf - -C "$dest"

        cp "${BASH_SOURCE[0]}" "${workdir}/deploy.sh" 2>/dev/null || true
        chmod +x "${workdir}/deploy.sh" 2>/dev/null || true
        return 0
    fi

    if [[ -d "${dest}/.git" ]]; then
        require_cmd git
        info "Updating source from ${SOURCE_REPO_URL} (${SOURCE_REPO_BRANCH}) ..."
        git -C "$dest" fetch --depth 1 origin "$SOURCE_REPO_BRANCH" || return 1
        git -C "$dest" checkout -f FETCH_HEAD || return 1
        cp "${BASH_SOURCE[0]}" "${workdir}/deploy.sh" 2>/dev/null || true
        chmod +x "${workdir}/deploy.sh" 2>/dev/null || true
        return 0
    fi

    if [[ -f "${dest}/Dockerfile" && -f "${dest}/package.json" ]]; then
        warn "Project root not found; using existing source at ${dest}."
        return 0
    fi

    require_cmd git
    info "No local source found; cloning ${SOURCE_REPO_URL} (${SOURCE_REPO_BRANCH}) to ${dest} ..."
    rm -rf "$dest"
    git clone --depth 1 --branch "$SOURCE_REPO_BRANCH" "$SOURCE_REPO_URL" "$dest" || return 1
    cp "${BASH_SOURCE[0]}" "${workdir}/deploy.sh" 2>/dev/null || true
    chmod +x "${workdir}/deploy.sh" 2>/dev/null || true
    return 0
}

generate_admin_password() {
    ADMIN_PASS="$(openssl rand -hex 12)"
}

write_pwd_file() {
    local workdir="$1"
    mkdir -p "${workdir}/configs"
    echo -n "$ADMIN_PASS" > "${workdir}/configs/pwd"
    chmod 600 "${workdir}/configs/pwd"
}

read_pwd_file() {
    local workdir="$1"
    if [[ -f "${workdir}/configs/pwd" ]]; then
        cat "${workdir}/configs/pwd"
    else
        echo "未能读取"
    fi
}

read_env_value() {
    local file="$1"
    local key="$2"
    grep -E "^${key}=" "$file" 2>/dev/null | tail -n 1 | cut -d= -f2- || true
}



create_env_file() {
    local workdir="$1"
    local host_port="$2"

    cat > "${workdir}/.env" <<EOF
PORT=${host_port}
TZ=Asia/Shanghai
GEMINI_OAUTH_CLIENT_ID=${GEMINI_OAUTH_CLIENT_ID:-}
GEMINI_OAUTH_CLIENT_SECRET=${GEMINI_OAUTH_CLIENT_SECRET:-}
ANTIGRAVITY_OAUTH_CLIENT_ID=${ANTIGRAVITY_OAUTH_CLIENT_ID:-}
ANTIGRAVITY_OAUTH_CLIENT_SECRET=${ANTIGRAVITY_OAUTH_CLIENT_SECRET:-}
ARGS=
EOF

    chmod 600 "${workdir}/.env"
}

create_default_config() {
    local workdir="$1"
    local config_file="${workdir}/configs/config.json"

    [[ -f "$config_file" ]] && return

    cat > "$config_file" <<'EOF'
{
  "REQUIRED_API_KEY": "123456",
  "SERVER_PORT": 3000,
  "HOST": "0.0.0.0",
  "MODEL_PROVIDER": "gemini-cli-oauth",
  "SYSTEM_PROMPT_FILE_PATH": null,
  "SYSTEM_PROMPT_MODE": "append",
  "SYSTEM_PROMPT_REPLACEMENTS": [],
  "PROMPT_LOG_BASE_NAME": "prompt_log",
  "PROMPT_LOG_MODE": "none",
  "REQUEST_MAX_RETRIES": 3,
  "REQUEST_BASE_DELAY": 1000,
  "CREDENTIAL_SWITCH_MAX_RETRIES": 5,
  "RATE_LIMIT_COOLDOWN_ENABLED": false,
  "RATE_LIMIT_COOLDOWN_MS": 30000,
  "RATE_LIMIT_COOLDOWN_JITTER_MS": 5000,
  "RATE_LIMIT_COOLDOWN_MAX_MS": 300000,
  "CRON_NEAR_MINUTES": 1,
  "CRON_REFRESH_TOKEN": false,
  "PROVIDER_POOLS_FILE_PATH": "configs/provider_pools.json",
  "CUSTOM_MODELS_FILE_PATH": "configs/custom_models.json",
  "MAX_ERROR_COUNT": 3,
  "providerFallbackChain": {},
  "modelFallbackMapping": {},
  "LOG_ENABLED": true,
  "LOG_OUTPUT_MODE": "all",
  "LOG_LEVEL": "info",
  "LOG_DIR": "logs",
  "LOG_INCLUDE_REQUEST_ID": true,
  "LOG_INCLUDE_TIMESTAMP": true,
  "LOG_MAX_FILE_SIZE": 10485760,
  "LOG_MAX_FILES": 10,
  "TLS_SIDECAR_ENABLED": false,
  "TLS_SIDECAR_PORT": 9090,
  "UI_ENABLED": true
}
EOF
}

create_compose_file() {
    local workdir="$1"

    cat > "${workdir}/docker-compose.yml" <<EOF
services:
  ${SERVICE_NAME}:
    build:
      context: ./${SOURCE_DIR_NAME}
      dockerfile: Dockerfile
    image: ${IMAGE_NAME}
    container_name: ${CONTAINER_NAME}
    restart: unless-stopped
    ports:
      - "\${PORT}:3000"
      - "${GEMINI_PORT_RANGE}:8085-8087"
      - "${CODEX_PORT}:1455"
      - "${KIRO_PORT_RANGE}:19876-19880"
    volumes:
      - ./configs:/app/configs
    environment:
      - TZ=\${TZ}
      - ARGS=\${ARGS}
      - GEMINI_OAUTH_CLIENT_ID=\${GEMINI_OAUTH_CLIENT_ID}
      - GEMINI_OAUTH_CLIENT_SECRET=\${GEMINI_OAUTH_CLIENT_SECRET}
      - ANTIGRAVITY_OAUTH_CLIENT_ID=\${ANTIGRAVITY_OAUTH_CLIENT_ID}
      - ANTIGRAVITY_OAUTH_CLIENT_SECRET=\${ANTIGRAVITY_OAUTH_CLIENT_SECRET}
    healthcheck:
      test: ["CMD", "node", "healthcheck.js"]
      interval: 30s
      timeout: 3s
      start_period: 5s
      retries: 3
EOF
}

show_access() {
    local workdir="$1"
    local env_file="${workdir}/.env"
    local host_port
    host_port="$(read_env_value "$env_file" PORT)"
    host_port="${host_port:-$DEFAULT_WEB_PORT}"

    local current_pass
    current_pass="$(read_pwd_file "$workdir")"

    echo ""
    echo "=================================================="
    echo -e "\033[32m✅ ${APP_NAME} 实例就绪\033[0m"
    echo "--------------------------------------------------"
    echo -e "Web 控制台: \033[36mhttp://$(get_local_ip):${host_port}\033[0m"
    echo "--------------------------------------------------"
    echo -e "后台密码: \033[31m${current_pass}\033[0m"
    echo -e "密码文件: \033[33m${workdir}/configs/pwd\033[0m"
    echo -e "配置目录: \033[33m${workdir}/configs\033[0m"
    echo -e "环境文件: \033[33m${workdir}/.env\033[0m"
    echo "--------------------------------------------------"
    echo "端口映射:"
    echo "  Web/API: ${host_port} -> 3000"
    echo "  Gemini/Antigravity OAuth: ${GEMINI_PORT_RANGE} -> 8085-8087"
    echo "  Codex OAuth: ${CODEX_PORT} -> 1455"
    echo "  Kiro OAuth: ${KIRO_PORT_RANGE} -> 19876-19880"
    echo "=================================================="
    echo ""
}

wait_app_ready() {
    info "等待 ${APP_NAME} 初始化..."

    for _ in $(seq 1 60); do
        if docker ps --format '{{.Names}} {{.Status}}' | grep -q "^${CONTAINER_NAME} .*Up"; then
            info "${APP_NAME} 已启动"
            return 0
        fi
        sleep 2
    done

    warn "${APP_NAME} 可能未正常启动，最近日志如下:"
    docker logs --tail=100 "$CONTAINER_NAME" 2>/dev/null || true
    return 1
}

deploy_aiclient2api() {
    info "== Starting ${APP_NAME} deployment =="

    require_cmd docker
    require_cmd awk
    require_cmd openssl
    require_cmd tar
    require_cmd git

    local dc_cmd
    dc_cmd="$(docker_compose_cmd)"

    read -r -p "Install path [default: ${DEFAULT_INSTALL_PATH}]: " input_path
    local install_path="${input_path:-$DEFAULT_INSTALL_PATH}"

    if [[ -d "$install_path" && "$(ls -A "$install_path" 2>/dev/null)" ]]; then
        err "Install path already contains data: ${install_path}"
        err "Choose another path, or run [8] uninstall first."
        return
    fi

    mkdir -p "$install_path"
    echo "$install_path" > "$ENV_RECORD_FILE"

    read -r -p "Web/API public port [default: ${DEFAULT_WEB_PORT}]: " input_port
    local host_port="${input_port:-$DEFAULT_WEB_PORT}"
    valid_port "$host_port" || die "Invalid port, must be 1-65535"
    if port_in_use "$host_port"; then
        local old_port="$host_port"
        host_port="$(find_free_port "$host_port")" || die "No free Web/API port found"
        warn "Web/API port ${old_port} is in use, using ${host_port} instead."
    fi
    resolve_oauth_ports

    mkdir -p "${install_path}/configs" "${install_path}/backups"
    chmod 700 "${install_path}/configs"
    chmod 755 "${install_path}/backups"

    generate_admin_password
    write_pwd_file "$install_path"
    create_env_file "$install_path" "$host_port"
    create_default_config "$install_path"
    sync_project_source "$install_path" || die "Project source not found. Run this script from project root, or set PROJECT_ROOT=/path/to/project."
    create_compose_file "$install_path"

    cd "$install_path" || return

    info "Building local image from current source and starting ${APP_NAME} ..."
    $dc_cmd build || die "Local image build failed"
    $dc_cmd up -d || die "Container start failed"

    wait_app_ready || true
    show_access "$install_path"
}

upgrade_service() {
    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && { err "No deployment found. Run [1] deploy first."; return; }

    cd "$workdir" || return
    local dc_cmd
    dc_cmd="$(docker_compose_cmd)"

    sync_project_source "$workdir" || die "Project source not found. Run this script from project root, or set PROJECT_ROOT=/path/to/project."
    create_compose_file "$workdir"

    info "Rebuilding container from current source ..."
    $dc_cmd build || die "Local image build failed"
    $dc_cmd up -d || die "Service start failed"

    wait_app_ready || true
    show_access "$workdir"
}

pause_service() {
    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && { err "未检测到部署环境。"; return; }

    cd "$workdir" || return
    $(docker_compose_cmd) stop
    info "服务已停止。"
}

restart_service() {
    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && { err "未检测到部署环境。"; return; }

    cd "$workdir" || return
    $(docker_compose_cmd) restart
    wait_app_ready || true
    show_access "$workdir"
}

reset_admin_password() {
    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && { err "未检测到部署环境。"; return; }

    generate_admin_password
    write_pwd_file "$workdir"

    cd "$workdir" || return
    $(docker_compose_cmd) restart "$SERVICE_NAME" >/dev/null 2>&1 || true

    info "后台密码已重置。"
    show_access "$workdir"
}

do_backup() {
    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && { err "No deployment found."; return; }

    local backup_dir="${workdir}/backups"
    mkdir -p "$backup_dir"

    local timestamp
    timestamp="$(date +"%Y%m%d_%H%M%S")"

    local temp_dir="${backup_dir}/tmp_${timestamp}"
    mkdir -p "$temp_dir"

    cp "${workdir}/docker-compose.yml" "${temp_dir}/" 2>/dev/null || true
    cp "${workdir}/.env" "${temp_dir}/" 2>/dev/null || true
    [[ -f "${workdir}/deploy.sh" ]] && cp "${workdir}/deploy.sh" "${temp_dir}/deploy.sh" 2>/dev/null || true
    [[ -d "${workdir}/configs" ]] && cp -a "${workdir}/configs" "${temp_dir}/configs"
    [[ -d "${workdir}/${SOURCE_DIR_NAME}" ]] && cp -a "${workdir}/${SOURCE_DIR_NAME}" "${temp_dir}/${SOURCE_DIR_NAME}"

    local backup_file="${backup_dir}/aiclient2api_backup_${timestamp}.tar.gz"

    tar -czf "$backup_file" -C "$temp_dir" .
    rm -rf "$temp_dir"

    find "$backup_dir" -maxdepth 1 -name 'aiclient2api_backup_*.tar.gz' -type f \
        | sort -r \
        | awk 'NR>5' \
        | xargs -r rm -f

    info "Backup completed: ${backup_file}"
}

restore_backup() {
    local workdir
    workdir="$(get_workdir)"

    local search_dir="${workdir:-$DEFAULT_INSTALL_PATH}/backups"
    local default_backup
    default_backup="$(ls -t "${search_dir}"/aiclient2api_backup_*.tar.gz 2>/dev/null | head -n 1 || true)"

    read -r -p "Backup file path [Enter for default: ${default_backup}]: " backup_path
    local path="${backup_path:-$default_backup}"
    [[ ! -f "$path" ]] && { err "No valid backup file found."; return; }

    local safe_backup="/tmp/$(basename "$path")"
    cp "$path" "$safe_backup" || die "Failed to copy backup file to temp directory"

    read -r -p "Restore target path [default: ${DEFAULT_INSTALL_PATH}]: " target_dir
    local wd="${target_dir:-$DEFAULT_INSTALL_PATH}"

    if [[ -d "$wd" ]]; then
        read -r -p "Target directory exists, overwrite? (y/N): " confirm
        [[ ! "$confirm" =~ ^[Yy]$ ]] && { rm -f "$safe_backup"; return; }

        cd "$wd" 2>/dev/null && $(docker_compose_cmd) down 2>/dev/null || true
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
        rm -rf "$wd"
    fi

    mkdir -p "$wd"
    tar -xzf "$safe_backup" -C "$wd" || die "Failed to extract backup"
    mkdir -p "${wd}/backups"
    cp "$safe_backup" "${wd}/backups/$(basename "$safe_backup")" 2>/dev/null || true
    rm -f "$safe_backup"

    echo "$wd" > "$ENV_RECORD_FILE"

    if [[ ! -f "${wd}/docker-compose.yml" || ! -f "${wd}/.env" ]]; then
        if port_in_use "$DEFAULT_WEB_PORT"; then
            DEFAULT_WEB_PORT="$(find_free_port "$DEFAULT_WEB_PORT")" || die "No free Web/API port found"
        fi
        resolve_oauth_ports
    fi
    [[ -f "${wd}/docker-compose.yml" ]] || create_compose_file "$wd"
    [[ -f "${wd}/.env" ]] || create_env_file "$wd" "$DEFAULT_WEB_PORT"
    [[ -f "${wd}/configs/pwd" ]] || { generate_admin_password; write_pwd_file "$wd"; }
    create_default_config "$wd"
    sync_project_source "$wd" || die "Backup has no source and current project root was not found; cannot build."

    cd "$wd" || return
    $(docker_compose_cmd) build || die "Local image build failed"
    $(docker_compose_cmd) up -d || die "Container start failed"

    wait_app_ready || true
    show_access "$wd"
}

setup_auto_backup() {
    require_cmd crontab

    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && { err "未检测到部署环境。"; return; }

    local cron_script="${workdir}/cron_backup.sh"
    local script_path
    script_path="$(readlink -f "${BASH_SOURCE[0]}")"

    echo " 1) 按固定分钟步进备份（推荐：10/15/20/30/60）"
    echo " 2) 按每日固定时间点备份（例如：每天 04:30）"
    echo " 3) 删除当前定时备份任务"

    read -r -p "请选择策略 [1/2/3]: " cron_type
    local cron_spec=""

    case "$cron_type" in
        1)
            read -r -p "请输入间隔分钟数: " min_interval
            [[ "$min_interval" =~ ^[0-9]+$ && "$min_interval" -ge 1 && "$min_interval" -le 1440 ]] || { err "分钟数无效"; return; }
            cron_spec="*/${min_interval} * * * *"
        ;;
        2)
            read -r -p "请输入每天固定备份时间 (HH:MM): " cron_time
            local hour="${cron_time%:*}"
            local minute="${cron_time#*:}"
            [[ "$hour" =~ ^[0-9]+$ && "$minute" =~ ^[0-9]+$ && "$hour" -le 23 && "$minute" -le 59 ]] || { err "时间格式无效"; return; }
            cron_spec="${minute} ${hour} * * *"
        ;;
        3)
            crontab -l 2>/dev/null | sed "/${CRON_TAG_BEGIN}/,/${CRON_TAG_END}/d" | crontab - 2>/dev/null || true
            rm -f "$cron_script"
            info "定时备份任务已删除。"
            return
        ;;
        *)
            err "无效选择"
            return
        ;;
    esac

    cat > "$cron_script" <<EOF
#!/usr/bin/env bash
bash "$script_path" run-backup >> "$BACKUP_LOG" 2>&1
EOF
    chmod +x "$cron_script"

    (
        crontab -l 2>/dev/null | sed "/${CRON_TAG_BEGIN}/,/${CRON_TAG_END}/d"
        echo "$CRON_TAG_BEGIN"
        echo "${cron_spec} bash ${cron_script}"
        echo "$CRON_TAG_END"
    ) | crontab -

    info "新的定时备份任务已注入。"
}

uninstall_service() {
    local workdir
    workdir="$(get_workdir)"
    [[ -z "$workdir" ]] && workdir="$DEFAULT_INSTALL_PATH"

    echo -e "\033[31m⚠️ 警告：这将彻底删除容器和本地配置数据！\033[0m"
    read -r -p "确认完全卸载？(y/N): " confirm
    [[ ! "$confirm" =~ ^[Yy]$ ]] && return

    if [[ -d "$workdir" ]]; then
        cd "$workdir" 2>/dev/null && $(docker_compose_cmd) down 2>/dev/null || true
    fi

    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker network rm aiclient2api_default 2>/dev/null || true
    rm -rf "$workdir"
    rm -f "$ENV_RECORD_FILE"
    crontab -l 2>/dev/null | sed "/${CRON_TAG_BEGIN}/,/${CRON_TAG_END}/d" | crontab - 2>/dev/null || true

    info "容器及配置数据已清理。"
}

install_ftp() {
    require_cmd curl
    clear
    echo -e "\033[32m📂 FTP/SFTP 备份工具...\033[0m"
    bash <(curl -L https://raw.githubusercontent.com/hiapb/ftp/main/back.sh)
}

main_menu() {
    clear
    echo "==================================================="
    echo "                ${APP_NAME} 一键管理"
    echo "==================================================="
    local wd
    wd="$(get_workdir)"
    echo -e " 实例运行路径: \033[36m${wd:-未部署}\033[0m"
    echo "---------------------------------------------------"
    echo "  1) 一键部署"
    echo "  2) 升级服务"
    echo "  3) 停止服务"
    echo "  4) 重启服务"
    echo "  5) 手动备份"
    echo "  6) 恢复备份"
    echo "  7) 定时备份"
    echo "  8) 完全卸载"
    echo "  9) FTP/SFTP 备份工具"
    echo " 10) 重置后台密码"
    echo "  0) 退出脚本"
    echo "==================================================="
    read -r -p "请输入操作序号 [0-10]: " choice

    case "$choice" in
        1) deploy_aiclient2api ;;
        2) upgrade_service ;;
        3) pause_service ;;
        4) restart_service ;;
        5) do_backup ;;
        6) restore_backup ;;
        7) setup_auto_backup ;;
        8) uninstall_service ;;
        9) install_ftp ;;
        10) reset_admin_password ;;
        0) info "欢迎下次使用，再见!"; exit 0 ;;
        *) warn "无效指令，请重新输入。" ;;
    esac
}

if [[ "${1:-}" == "run-backup" ]]; then
    do_backup
else
    if [[ $EUID -ne 0 ]]; then
        die "请使用 root 权限执行脚本，例如: sudo bash deploy.sh"
    fi

    while true; do
        main_menu
        echo ""
        read -r -p "➤ 按回车键返回主菜单..."
    done
fi
