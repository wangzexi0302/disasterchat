import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from app.api.database.db_setup import get_db

try:
    db = next(get_db())
    print("数据库连接成功！")
except Exception as e:
    print(f"数据库连接失败：{e}")
finally:
    if 'db' in locals():
        db.close()