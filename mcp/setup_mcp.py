# -*- coding: utf-8 -*-
"""一键安装脚本：注册 MCP Server + 导入技能包.

用法:
    python mcp/setup_mcp.py

功能:
    1. 自动检测当前 Python 解释器路径和项目根目录
    2. 将 captcha MCP Server 写入 ~/.workbuddy/mcp.json
    3. 将 captcha-recognition 技能包复制到 ~/.workbuddy/skills/
    4. 输出后续操作指引
"""
import json
import os
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKBUDDY_HOME = os.path.expanduser("~/.workbuddy")
MCP_JSON = os.path.join(WORKBUDDY_HOME, "mcp.json")
SKILLS_DIR = os.path.join(WORKBUDDY_HOME, "skills")
SKILL_SRC = os.path.join(PROJECT_DIR, "mcp", "captcha-recognition")
SKILL_DST = os.path.join(SKILLS_DIR, "captcha-recognition")


def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")


def register_mcp():
    """将 captcha MCP Server 写入 mcp.json."""
    step("[1/3] 注册 MCP Server")

    entry = {
        "command": sys.executable,
        "args": [os.path.join(PROJECT_DIR, "mcp", "mcp_server.py")],
        "cwd": PROJECT_DIR,
        "disabled": False,
    }

    # 读取现有配置，合并 captcha 条目
    if os.path.exists(MCP_JSON):
        with open(MCP_JSON, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["captcha"] = entry

    os.makedirs(WORKBUDDY_HOME, exist_ok=True)
    with open(MCP_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Python : {sys.executable}")
    print(f"  Server : {entry['args'][0]}")
    print(f"  配置文件: {MCP_JSON}")
    print("  ✓ MCP Server 已注册")


def install_skill():
    """复制技能包到 ~/.workbuddy/skills/."""
    step("[2/3] 导入技能包")

    if not os.path.isdir(SKILL_SRC):
        print(f"  ⚠ 技能包目录不存在: {SKILL_SRC}")
        print(f"  跳过技能导入（MCP Server 仍可正常使用）")
        return

    os.makedirs(SKILLS_DIR, exist_ok=True)

    # 如果已存在先删除再复制
    if os.path.exists(SKILL_DST):
        shutil.rmtree(SKILL_DST)

    shutil.copytree(SKILL_SRC, SKILL_DST)
    print(f"  来源  : {SKILL_SRC}")
    print(f"  目标  : {SKILL_DST}")
    print("  ✓ 技能包已导入")


def verify():
    """验证安装."""
    step("[3/3] 验证安装")

    # 检查 mcp_server.py 可导入
    mcp_server = os.path.join(PROJECT_DIR, "mcp", "mcp_server.py")
    if os.path.exists(mcp_server):
        print(f"  ✓ mcp_server.py 存在")
    else:
        print(f"  ✗ mcp_server.py 不存在!")
        return

    # 检查 mcp.json
    if os.path.exists(MCP_JSON):
        with open(MCP_JSON, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "captcha" in config.get("mcpServers", {}):
            print(f"  ✓ mcp.json 已注册 captcha 服务")
        else:
            print(f"  ✗ mcp.json 中未找到 captcha 服务")
    else:
        print(f"  ✗ mcp.json 不存在")

    # 检查技能包
    skill_md = os.path.join(SKILL_DST, "SKILL.md")
    if os.path.exists(skill_md):
        print(f"  ✓ 技能包 captcha-recognition 已就位")
    else:
        print(f"  ⚠ 技能包未导入（不影响 MCP Server 使用）")


def main():
    print("=" * 50)
    print("  captcha_alpha 一键安装")
    print("=" * 50)
    print(f"  项目目录: {PROJECT_DIR}")
    print(f"  Python  : {sys.executable}")

    register_mcp()
    install_skill()
    verify()

    step("完成！下一步操作")
    print("""
  1. 重启 WorkBuddy（或重新加载连接器）
  2. 打开 WorkBuddy → 右上角连接器管理
  3. 找到 captcha → 点击「Trust」启用
  4. 在对话中说「识别验证码 /path/to/captcha.png」即可调用

  命令行直接使用:
    python src/main.py images/test.png      # → xf4y4
    python src/main.py images/test2.jpg     # → kdqu
    pytest tests/                           # 运行测试套件
""")


if __name__ == "__main__":
    main()
