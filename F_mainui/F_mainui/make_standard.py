import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import yaml

_PROJECT_ROOT = os.path.join(_REPO_ROOT)
_RPTCFG_PATH = os.path.join(_PROJECT_ROOT, "config", "rptcfg.yaml")
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
_OUT_TABLE_JSON = os.path.join(_PROJECT_ROOT, "table.json")

# 统一从 config/rptcfg.yaml 读取标准配置
with open(_RPTCFG_PATH, "r", encoding="utf-8") as f:
    rptcfg = yaml.safe_load(f) or {}

product_cls = str(rptcfg.get("product_cls", "1"))
class_labels = rptcfg.get("class_labels", {})  # {id:int -> name:str}
class_list = rptcfg.get("class_list", [])      # [id,...]
data = rptcfg.get(f"data{product_cls}", [])

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
area_ranges = config.get("anomaly_area_cls_range", [])

# 行名：按 class_list 顺序取中文名（找不到则用 id 字符串兜底）
index_list = []
for cls in class_list:
    try:
        cls_int = int(cls)
    except Exception:
        cls_int = cls
    index_list.append(str(class_labels.get(cls_int, cls_int)))

# 列名：与报告统计表完全一致（来自 anomaly_area_cls_range）
if not area_ranges:
    raise SystemExit("[ERROR] config.yaml 中 anomaly_area_cls_range 为空，无法生成标准表。")
max_end = area_ranges[-1][1]
columns_list = [f"{start}-{end}" for start, end in area_ranges] + [f"> {max_end}"]

# DataFrame：行=类别，列=面积区间，值=允许的最大数量
table = pd.DataFrame(data, index=index_list, columns=columns_list)

# 方案1：写入统一标准文件 table.json（报告生成/盖章判定读取它）
table.to_json(_OUT_TABLE_JSON, orient="split", index=True, force_ascii=False)
print(f"判定标准已写入：{_OUT_TABLE_JSON} (product_cls={product_cls})")