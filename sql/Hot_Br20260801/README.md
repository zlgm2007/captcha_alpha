# SQL 目录（分支 Hot_Br20260801）

本需求（验证码工作台集成 + 模型训练页面）**不涉及数据库**：

- `captcha_data_labeler/`：纯文件系统（raw/labeled/unrecognizable 目录 + 文件改名），无数据库。
- `captcha_data_recognizer/`：纯文件系统 + 内存任务表（批跑 job 存于进程内 dict）。
- `captcha_trainer/`：训练数据走 cache 文件（cache.train.tmp / cache.val.tmp）+ config.yaml。

因此本分支无需 SQL 脚本。若后续引入数据库，SQL 文件放在本目录。
