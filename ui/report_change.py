from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QMessageBox
from PyQt5.QtWidgets import (
    QTableWidget,
    QHBoxLayout,
    QApplication,
    QWidget,
    QLineEdit,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QRect
from report import Ui_Report  # 引用生成的 ui_para.py 文件
from cls_config import ClsConfigWindow
import ast
import yaml
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import subprocess

# ── 路径常量（与 main.py 保持一致）──────────────────────────────────────────
_PYTHON_EXE = os.environ.get('STEEL_PYTHON_EXE', sys.executable)
_GEN_REPORT_MODULE = "app.report.gen_report_cls"
_MAKE_STD_SCRIPT   = os.path.join(_REPO_ROOT, "ui", "make_standard.py")
_DETECT_RESULT_DIR = os.path.join(_REPO_ROOT, "detect result")
# ────────────────────────────────────────────────────────────────────────────


class ReportWindow(QMainWindow, Ui_Report):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("报告打印与标准维护")
        self.class_counter = 1  # 用于自动分配 class_id
        self.input_fields = []
        self.config_data = {}
        self.updates_up = []
        self.updates_down = []

        self.config_data['update_info'] = True

        self.justsaveone('update_info', True)

        self.initUI1()
        self.initUI2()
        self.initUI3()

        self.all_ok.clicked.connect(self.print_report)
        self.sdandard_show.clicked.connect(self.standard_show_click)
        # 旧版“报告定位”功能已由报告中心统一接管，这里不再暴露入口
        # self.pushButton_id.clicked.connect(self.pushButton_id_click)

        self._apply_compact_layout()
        self._sync_selection_labels()
        self._polish_report_spacing()

    def _sync_selection_labels(self):
        """从 rptcfg.yaml 同步当前选择信息到界面，同时预填打印参数输入框。"""
        try:
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        try:
            _time = str(cfg.get("time", ""))
            _id   = str(cfg.get("id", ""))
            _strip = str(cfg.get("strip_id", ""))
            # show_* 标签（现已隐藏，但保留赋值兼容旧逻辑）
            if hasattr(self, "show_time"):
                self.show_time.setText(_time)
            if hasattr(self, "show_id"):
                self.show_id.setText(_id)
            if hasattr(self, "show_cls"):
                self.show_cls.setText(str(cfg.get("check_cls", "")))
            if hasattr(self, "show_strip_id"):
                self.show_strip_id.setText(_strip)
            # 将值预填到可见的打印参数输入框
            if hasattr(self, "time") and not self.time.text():
                self.time.setText(_time)
            if hasattr(self, "id") and not self.id.text():
                self.id.setText(_id)
            if hasattr(self, "strip_id") and not self.strip_id.text():
                self.strip_id.setText(_strip)
        except Exception:
            pass

    def refresh_from_report_center(self):
        """
        报告中心选择后进入「修改界面」时调用：强制从 rptcfg.yaml 刷新左上角显示，
        并同步产品型号下拉框与标准表格，避免复用旧窗口导致信息不一致。
        """
        self._sync_selection_labels()
        # 同步产品型号下拉框选择（若存在）
        try:
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        try:
            key = str(cfg.get("product_cls", "")).strip()
            if hasattr(self, "product_cls"):
                self.product_cls.setText(key)
            if hasattr(self, "product_cls_combo"):
                ix = -1
                for i in range(self.product_cls_combo.count()):
                    if str(self.product_cls_combo.itemData(i)) == key:
                        ix = i
                        break
                if ix >= 0 and self.product_cls_combo.currentIndex() != ix:
                    self.product_cls_combo.setCurrentIndex(ix)
        except Exception:
            pass
        # 刷新允收矩阵
        try:
            self.create_table()
        except Exception:
            pass

    def _apply_compact_layout(self):
        """
        将左上角"报告定位"frame 改造成紧凑的"打印参数"条（日期/生产卡号/钢带号），
        隐藏其中的报告定位导航元素，保留并重新布局三个关键输入字段，方便用户在打印前
        确认参数。其余控件相应下移。
        """
        _R = QRect

        # 1) 改造 frame：隐藏定位导航装饰，只保留打印参数输入对
        if hasattr(self, "frame"):
            # 不隐藏 show_time/show_id/show_strip_id：你希望左上角直接固定显示报告中心确定的内容
            for attr in ("label_2", "pushButton_id", "show_cls",
                         "label_5", "check_cls", "label_13"):
                if hasattr(self, attr):
                    getattr(self, attr).hide()

            # 标签预留足够宽度，输入框与标签之间留间隙，避免「别」「号」等被遮挡
            if hasattr(self, "label_3"):
                self.label_3.setGeometry(_R(12, 10, 52, 26))
            if hasattr(self, "time"):
                self.time.setGeometry(_R(72, 10, 132, 26))
            if hasattr(self, "label_4"):
                self.label_4.setGeometry(_R(12, 42, 88, 26))
            if hasattr(self, "id"):
                self.id.setGeometry(_R(108, 42, 122, 26))
            if hasattr(self, "label_14"):
                self.label_14.setGeometry(_R(240, 42, 56, 26))
            if hasattr(self, "strip_id"):
                self.strip_id.setGeometry(_R(304, 42, 84, 26))

            # 按你的要求：左上角日期/卡号/钢带号不再使用输入框（由报告中心确定）
            if hasattr(self, "time"):
                self.time.hide()
            if hasattr(self, "id"):
                self.id.hide()
            if hasattr(self, "strip_id"):
                self.strip_id.hide()
            if hasattr(self, "pushButton_id"):
                self.pushButton_id.hide()

            # 显示只读标签，并放到原输入框的位置
            if hasattr(self, "show_time"):
                self.show_time.setGeometry(_R(72, 10, 132, 26))
                self.show_time.show()
            if hasattr(self, "show_id"):
                self.show_id.setGeometry(_R(108, 42, 122, 26))
                self.show_id.show()
            if hasattr(self, "show_strip_id"):
                self.show_strip_id.setGeometry(_R(304, 42, 84, 26))
                self.show_strip_id.show()

            self.frame.setGeometry(_R(40, 0, 420, 76))
            self.frame.setToolTip(
                "打印报告所需的基本参数。\n"
                "通过「报告生成」入口选卷后会自动填充；本界面不再提供手动输入。"
            )
            self.frame.show()

        # 2) 缺陷类别区域整体下移，为打印参数条腾出空间
        try:
            if hasattr(self, "label"):
                self.label.setGeometry(_R(50, 82, 220, 28))
        except Exception:
            pass

        try:
            if hasattr(self, "scrollArea01"):
                self.scrollArea01.setGeometry(_R(40, 114, 400, 692))
        except Exception:
            pass

        try:
            if hasattr(self, "pushButton_add"):
                self.pushButton_add.setGeometry(_R(60, 820, 151, 41))
            if hasattr(self, "cls_save"):
                self.cls_save.setGeometry(_R(270, 820, 151, 41))
            if hasattr(self, "all_ok"):
                self.all_ok.setGeometry(_R(60, 873, 361, 51))
        except Exception:
            pass

        # 3) 右侧主区域左移，填补原 frame 区域的空白
        try:
            if hasattr(self, "frame_2"):
                self.frame_2.setGeometry(_R(440, 0, 900, 991))
        except Exception:
            pass

    def _polish_report_spacing(self):
        """微调标签与输入框间距，避免中文标签与控件边缘重叠。"""
        _R = QRect
        try:
            # 右侧：产品型号行（「产品类别」与输入框、按钮拉开）
            if hasattr(self, "label_12"):
                self.label_12.setGeometry(_R(20, 46, 118, 28))
            if hasattr(self, "product_cls"):
                self.product_cls.setGeometry(_R(148, 46, 172, 28))
            # 产品型号下拉框（替代 product_cls 输入框）
            if hasattr(self, "product_cls_combo"):
                # 左移 + 加宽，避免与「查看」重叠；并与「查看」同高
                self.product_cls_combo.setGeometry(_R(148, 46, 220, 30))
                self.product_cls_combo.setMinimumHeight(30)
                self.product_cls_combo.setStyleSheet("QComboBox{padding:2px 10px;}")
            if hasattr(self, "sdandard_show"):
                # 与下拉框留出间距，并与下拉框同高
                self.sdandard_show.setGeometry(_R(380, 46, 72, 30))
            if hasattr(self, "_btn_open_cls_cfg"):
                self._btn_open_cls_cfg.setGeometry(460, 46, 118, 30)
            if hasattr(self, "standard_save"):
                self.standard_save.setGeometry(_R(688, 40, 151, 41))

            # 右侧：优化选项行（打印面积 / 打印类别）
            if hasattr(self, "label_8"):
                self.label_8.setGeometry(_R(20, 476, 100, 28))
            if hasattr(self, "lineEdit_area"):
                self.lineEdit_area.setGeometry(_R(128, 472, 172, 34))
                self.lineEdit_area.setMinimumHeight(34)
                self.lineEdit_area.setStyleSheet("QLineEdit{padding:2px 8px;}")
            if hasattr(self, "label_9"):
                self.label_9.setGeometry(_R(318, 476, 100, 28))
            if hasattr(self, "lineEdit_cls"):
                self.lineEdit_cls.setGeometry(_R(426, 472, 172, 34))
                self.lineEdit_cls.setMinimumHeight(34)
                self.lineEdit_cls.setStyleSheet("QLineEdit{padding:2px 8px;}")
            if hasattr(self, "change_save"):
                self.change_save.setGeometry(_R(688, 512, 151, 41))

            # Tab 内标题大致居中
            if hasattr(self, "label_10"):
                self.label_10.setGeometry(_R(270, 8, 200, 28))
            if hasattr(self, "label_11"):
                self.label_11.setGeometry(_R(270, 8, 200, 28))
        except Exception:
            pass

    def _apply_table_spacing(self):
        """表头/行标题增加内边距与最小宽度，避免文字贴边。"""
        _tbl_style = (
            "QTableWidget { gridline-color: #c8c8c8; }"
            "QHeaderView::section { padding: 6px 10px; min-height: 28px; }"
        )
        for tw in (
            getattr(self, "tableWidget_standard", None),
            getattr(self, "tableWidget_change", None),
            getattr(self, "tableWidget_change_2", None),
        ):
            if tw is None:
                continue
            tw.setStyleSheet(_tbl_style)
            vh = tw.verticalHeader()
            vh.setSectionResizeMode(QHeaderView.Fixed)
            vh.setMinimumWidth(92)
            vh.setDefaultAlignment(Qt.AlignCenter)
            hh = tw.horizontalHeader()
            hh.setDefaultAlignment(Qt.AlignCenter)
            hh.setMinimumSectionSize(72)

    def closeEvent(self, event):
        # 关闭窗口时将 update_info 置 False，表示本次报告会话结束；
        # 报告中心下次生成前会重新置 True 并写入当前选择。
        self.justsaveone('update_info', False)
        super().closeEvent(event)

    def print_report(self):
        self.non_blocking_information(self, "报告打印", "报告打印中，请等待。")
        try:
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as f:
                rptcfg = yaml.safe_load(f) or {}
        except Exception:
            rptcfg = {}
        time = (self.time.text() or "").strip() or str(rptcfg.get("time", ""))
        cid = (self.id.text() or "").strip() or str(rptcfg.get("id", ""))
        strip = (self.strip_id.text() or "").strip() or str(rptcfg.get("strip_id", "1"))
        if not time or not cid:
            QMessageBox.warning(
                self,
                "信息不全",
                "请先填写日期、生产卡号（或在上方确认已保存到配置），再打印报告。",
            )
            return
        try:
            strip_n = str(max(1, int(float(strip))))
        except (ValueError, TypeError):
            strip_n = "1"
        args = [
            _PYTHON_EXE,
            "-u",
            "-m",
            _GEN_REPORT_MODULE,
            "--date",
            time,
            "--conduct-id",
            cid,
            "--strip-id",
            strip_n,
            "--update-info",
            "true",
        ]
        self.python_process = subprocess.Popen(args, cwd=os.path.join(_REPO_ROOT))


    # 替代 QMessageBox.information 的非阻塞实现
    def non_blocking_information(self, parent, title, message):
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.show()  # 非阻塞显示
        return msg_box  # 返回消息框对象（如果需要进一步操作）

    def pushButton_id_click(self):
        self.config_data['update_info'] = True
        self.justsaveone('update_info', True)
        try:
            time = self.time.text()
            id = self.id.text()
            strip_id = self.strip_id.text()
            check_cls = self.check_cls.text()
            self.config_data['time'] = time
            self.config_data['id'] = id
            self.config_data['check_cls'] = check_cls
            self.config_data['strip_id'] = strip_id
            self.justsaveone('time', time)
            self.justsaveone('id', id)
            self.justsaveone('check_cls', check_cls)
            self.justsaveone('strip_id', strip_id)

            with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.show_time.setText(str(config['time']))
            self.show_id.setText(str(config['id']))
            self.show_cls.setText(str(config['check_cls']))
            self.show_strip_id.setText(str(config['strip_id']))
            result = self.find_folders_with_id(time, _DETECT_RESULT_DIR, id, strip_id)
            if result:
                self.find_and_open_image(result[0])
            else:
                QMessageBox.warning(self, "目录不存在",
                                    f"未找到报告目录：\n"
                                    f"  日期：{time}\n"
                                    f"  卡号：{id}\n"
                                    f"  钢带号：{strip_id}\n\n"
                                    "请确认已完成报告生成，且日期/卡号/钢带号与生成时一致。")
        except Exception as e:
            QMessageBox.information(self, "输入错误", "请输入正确的报告信息！")


    def find_and_open_image(self, folder_path):
        """按优先级搜索报告产物并用系统默认程序打开。"""
        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.warning(self, "目录不存在",
                                f"报告目录不存在：\n{folder_path}\n\n"
                                "请确认日期/生产卡号/钢带号输入正确，并已完成报告生成。")
            return

        # 按优先级依次尝试打开
        candidates = []
        # 1. 检测报告 PDF
        for fn in os.listdir(folder_path):
            if fn.endswith(".pdf"):
                candidates.append(os.path.join(folder_path, fn))
        # 2. 上表面分布图 PNG
        for fn in os.listdir(folder_path):
            if fn.endswith(".png") and "上表面" in fn and "distribution" in fn:
                candidates.append(os.path.join(folder_path, fn))
        # 3. 任意 PNG
        for fn in os.listdir(folder_path):
            if fn.endswith(".png"):
                candidates.append(os.path.join(folder_path, fn))

        if not candidates:
            QMessageBox.information(self, "暂无产物",
                                    f"目录内未找到报告文件（PDF/PNG）：\n{folder_path}")
            return

        target = candidates[0]
        try:
            os.startfile(target)
            print(f"已打开报告文件: {target}")
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"无法打开文件：\n{target}\n\n{e}")

    def find_folders_with_id(self, date_str, base_path, product_id, strip_id):
        """
        定位报告子目录：{base_path}\\{date_str}\\{product_id}\\report\\strip_{strip_id}
        与 gen_report_cls.py 生成的目录结构保持一致。

        :param date_str:   日期字符串，格式 YYYYMMDD
        :param base_path:  detect result 根目录
        :param product_id: 生产卡号
        :param strip_id:   钢带号（1 基准，字符串或整数均可）
        :return:           [path] 若存在，否则 []
        """
        strip_dir_name = f"strip_{strip_id}"
        target = os.path.join(base_path, date_str, product_id, "report", strip_dir_name)

        if os.path.exists(target):
            print(f"找到报告目录: {target}")
            return [target]

        print(f"报告目录不存在: {target}")
        return []

    def initUI3(self):
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.lineEdit_area.setText(str(config['area_range']))
        self.lineEdit_cls.setText(str(config['print_cls']))

        # 生成表格
        self.create_table2()
        self.create_table3()

        self.change_save.clicked.connect(self.change_data)

    def create_table2(self):

        # 获取行列数
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # 获取 class_labels 的最大 key 值
        class_labels = data.get('class_labels', {})
        self.row_headers = list(class_labels.values())
        self.row2 = max(map(int, class_labels.keys()), default=0)

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        anomaly_area_cls_range = config.get('anomaly_area_cls_range', [])
        last_end = anomaly_area_cls_range[-1][1]  # 获取最后一个范围的结束值
        self.column_headers = [f"[{start}, {end}]" for start, end in anomaly_area_cls_range] + [f"> {last_end}"]
        self.col2 = len(anomaly_area_cls_range)+1

        rows = self.row2
        cols = self.col2

        # 设置表格行列数
        self.tableWidget_change.setRowCount(rows)
        self.tableWidget_change.setColumnCount(cols)

        # 设置行标题和列标题
        self.tableWidget_change.setHorizontalHeaderLabels(self.column_headers)
        self.tableWidget_change.setVerticalHeaderLabels(self.row_headers)

        # 初始化表格内容
        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem("")  # 默认值为 空
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_change.setItem(i, j, item)

        # 调整列宽
        for col in range(cols):
            self.tableWidget_change.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(rows):
            self.tableWidget_change.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def create_table3(self):


        # 设置表格行列数
        self.tableWidget_change_2.setRowCount(self.row2)
        self.tableWidget_change_2.setColumnCount(self.col2)

        # 设置行标题和列标题
        self.tableWidget_change_2.setHorizontalHeaderLabels(self.column_headers)
        self.tableWidget_change_2.setVerticalHeaderLabels(self.row_headers)

        # 初始化表格内容
        for i in range(self.row2):
            for j in range(self.col2):
                item = QTableWidgetItem("")  # 默认值为 1000
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_change_2.setItem(i, j, item)

        # 调整列宽
        for col in range(self.col2):
            self.tableWidget_change_2.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(self.row2):
            self.tableWidget_change_2.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def change_data(self):
        self.updates_up = []
        self.updates_down = []
        try:
            area_range = ast.literal_eval(self.lineEdit_area.text().strip())
        except (ValueError, SyntaxError) as e:
            QMessageBox.warning(self, "输入格式错误",
                                f"打印面积格式错误，请输入合法的 Python 列表（如 [] 或 [100,500]）：\n{e}")
            return
        try:
            print_cls = ast.literal_eval(self.lineEdit_cls.text().strip())
        except (ValueError, SyntaxError) as e:
            QMessageBox.warning(self, "输入格式错误",
                                f"打印类别格式错误，请输入合法的 Python 列表（如 [1,2,3]）：\n{e}")
            return
        self.config_data['area_range'] = area_range
        self.config_data['print_cls'] = print_cls

        for i in range(self.row2):
            for j in range(self.col2):
                # 获取单元格数据
                item = self.tableWidget_change.item(i, j)
                if item:
                    text = item.text().strip()
                    if text.isdigit() or text == "0":  # 检查是否是数字
                        new_count = int(text)
                        if new_count >= 0:  # 忽略无效数据
                            entry = {
                                "class_label": self.row_headers[i],
                                "area_interval": self.column_headers[j],
                                "new_count": new_count,
                            }
                            self.updates_up.append(entry)
        self.config_data['updates_up'] = self.updates_up
        print(self.updates_up)

        for i in range(self.row2):
            for j in range(self.col2):
                # 获取单元格数据
                item2 = self.tableWidget_change_2.item(i, j)
                if item2:
                    text2 = item2.text().strip()
                    if text2.isdigit() or text2 == "0":  # 检查是否是数字
                        new_count2 = int(text2)
                        if new_count2 >= 0:  # 忽略无效数据
                            entry2 = {
                                "class_label": self.row_headers[i],
                                "area_interval": self.column_headers[j],
                                "new_count": new_count2,
                            }
                            self.updates_down.append(entry2)
        self.config_data['updates_down'] = self.updates_down
        print(self.updates_down)
        self.justsaveone('area_range', area_range)
        self.justsaveone('print_cls', print_cls)
        self.justsaveone('updates_up', self.updates_up)
        self.justsaveone('updates_down', self.updates_down)

    def initUI2(self):
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        key = str(config.get('product_cls', ''))
        # 合格标准中：产品类型改为下拉框选择（来自 rptcfg.yaml）
        try:
            from cls_config import product_combo_entries
            entries = product_combo_entries()
        except Exception:
            entries = []

        # 仍保留 product_cls QLineEdit 作为内部数据载体，但不展示给用户
        self.product_cls.setText(key)
        self.product_cls.setReadOnly(True)
        try:
            self.product_cls.setToolTip("产品型号（来自 rptcfg）。可在下方下拉框选择后刷新表格。")
        except Exception:
            pass
        self.product_cls.setStyleSheet("background: #f5f5f5; color: #555;")

        # 创建下拉框并映射到同一几何
        if not hasattr(self, "product_cls_combo"):
            self.product_cls_combo = QComboBox(self.frame_2)
        # 尺寸/位置由 _polish_report_spacing 统一管理；这里给一个安全的最小宽度兜底
        self.product_cls_combo.setMinimumWidth(220)
        self.product_cls_combo.setMinimumHeight(30)
        try:
            self.product_cls_combo.setStyleSheet("QComboBox{padding:2px 10px;}")
        except Exception:
            pass

        self.product_cls_combo.clear()
        for k, disp in entries:
            if not k:
                continue
            self.product_cls_combo.addItem(str(disp), str(k))

        # 设置初始选择
        ix = -1
        for i in range(self.product_cls_combo.count()):
            if str(self.product_cls_combo.itemData(i)) == key:
                ix = i
                break
        if ix >= 0:
            self.product_cls_combo.setCurrentIndex(ix)

        # 显示下拉框，隐藏原输入框
        self.product_cls_combo.show()
        self.product_cls.hide()

        def _on_cls_changed(_=None):
            selected_key = str(self.product_cls_combo.currentData() or "")
            if not selected_key:
                return
            self.product_cls.setText(selected_key)
            # 切换产品类型后刷新允收矩阵视图
            try:
                self.standard_show_click()
            except Exception:
                # 兜底：至少重建表格
                try:
                    self.create_table()
                except Exception:
                    pass

        # 避免重复连接
        try:
            self.product_cls_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.product_cls_combo.currentIndexChanged.connect(_on_cls_changed)

        # 「类别配置」快捷按钮（放在产品类别行旁边）
        if not hasattr(self, '_btn_open_cls_cfg'):
            from PyQt5.QtWidgets import QPushButton
            from PyQt5.QtGui import QFont
            self._btn_open_cls_cfg = QPushButton("类别配置 →", self.frame_2)
            self._btn_open_cls_cfg.setGeometry(360, 47, 100, 30)
            f = QFont("Arial", 10); f.setBold(True)
            self._btn_open_cls_cfg.setFont(f)
            self._btn_open_cls_cfg.setStyleSheet(
                "QPushButton{background:#E8F5E9;color:#2e7d32;"
                "border:1px solid #a5d6a7;border-radius:4px;}"
                "QPushButton:hover{background:#C8E6C9;}"
            )
            self._btn_open_cls_cfg.setToolTip(
                "打开类别配置窗口维护型号、缺陷类别和允收矩阵。"
            )
            self._btn_open_cls_cfg.clicked.connect(self._open_cls_config)
            self._btn_open_cls_cfg.show()
            self._cls_config_window = None

        # 生成表格（只读视图；修改请通过「类别配置」）
        self.create_table()
        from PyQt5.QtWidgets import QAbstractItemView
        self.tableWidget_standard.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget_standard.setToolTip(
            "当前型号的允收标准（只读）。\n"
            "如需修改请点击「类别配置 →」按钮，在类别配置窗口中统一维护。"
        )

        # 保留导出按钮但更新说明，提示优先走类别配置
        self.standard_save.clicked.connect(self.export_data)
        self.standard_save.setText("强制同步")
        self.standard_save.setToolTip(
            "将当前表格内容强制写回 rptcfg.yaml（紧急覆盖用途）。\n"
            "正常情况下请通过「类别配置」窗口维护，避免双份编辑。"
        )

    def standard_show_click(self):
        product_cls = self.product_cls.text()
        # 获取行列数
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # 获取 class_labels 的最大 key 值
        class_labels = data.get('class_labels', {})


        row_headers = list(class_labels.values())
        self.row = max(map(int, class_labels.keys()), default=0)

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        anomaly_area_cls_range = config.get('anomaly_area_cls_range', [])
        last_end = anomaly_area_cls_range[-1][1]  # 获取最后一个范围的结束值
        column_headers = [f"[{start}, {end}]" for start, end in anomaly_area_cls_range] + [f"> {last_end}"]
        self.col = len(anomaly_area_cls_range) + 1

        rows = self.row
        cols = self.col

        # 设置表格行列数
        self.tableWidget_standard.setRowCount(rows)
        self.tableWidget_standard.setColumnCount(cols)

        # 设置行标题和列标题
        self.tableWidget_standard.setHorizontalHeaderLabels(column_headers)
        self.tableWidget_standard.setVerticalHeaderLabels(row_headers)

        # 初始化表格内容
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            config2 = yaml.safe_load(file)
        data_key = f"data{product_cls}"
        data = config2.get(data_key)
        for i in range(rows):
            for j in range(cols):
                # 如果数据列表中有值，则使用该值；否则默认为 1000
                value = data[i][j] if i < len(data) and j < len(data[i]) else 1000
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_standard.setItem(i, j, item)

        # 调整列宽
        for col in range(cols):
            self.tableWidget_standard.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(rows):
            self.tableWidget_standard.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def create_table(self):
        # 获取行列数
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # 获取 class_labels 的最大 key 值
        class_labels = data.get('class_labels', {})
        product_cls = data.get("product_cls")


        row_headers = list(class_labels.values())
        self.row = max(map(int, class_labels.keys()), default=0)

        with open(os.path.join(_REPO_ROOT, 'config', 'config.yaml'), 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        anomaly_area_cls_range = config.get('anomaly_area_cls_range', [])
        last_end = anomaly_area_cls_range[-1][1]  # 获取最后一个范围的结束值
        column_headers = [f"[{start}, {end}]" for start, end in anomaly_area_cls_range] + [f"> {last_end}"]
        self.col = len(anomaly_area_cls_range)+1

        rows = self.row
        cols = self.col

        # 设置表格行列数
        self.tableWidget_standard.setRowCount(rows)
        self.tableWidget_standard.setColumnCount(cols)

        # 设置行标题和列标题
        self.tableWidget_standard.setHorizontalHeaderLabels(column_headers)
        self.tableWidget_standard.setVerticalHeaderLabels(row_headers)

        # 初始化表格内容
        with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as file:
            config2 = yaml.safe_load(file)
        data_key = f"data{product_cls}"
        data = config2.get(data_key)
        for i in range(rows):
            for j in range(cols):
                # 如果数据列表中有值，则使用该值；否则默认为 1000
                value = data[i][j] if i < len(data) and j < len(data[i]) else 1000
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.tableWidget_standard.setItem(i, j, item)

        # 调整列宽
        for col in range(cols):
            self.tableWidget_standard.setColumnWidth(col, 105)  # 每列宽度为 60 像素

        # 调整行高
        for row in range(rows):
            self.tableWidget_standard.setRowHeight(row, 40)  # 每行高度为 30 像素

        self._apply_table_spacing()

    def export_data(self):
        product_cls = self.product_cls.text()
        self.config_data['product_cls'] = product_cls
        rows = self.tableWidget_standard.rowCount()
        cols = self.tableWidget_standard.columnCount()

        # 获取表格数据
        data = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                item = self.tableWidget_standard.item(i, j)
                if item and item.text().strip().isdigit():
                    row_data.append(int(item.text().strip()))
                else:
                    row_data.append(0)  # 默认值为 0
            data.append(row_data)

        self.config_data[f"data{product_cls}"] = data
        self.justsaveone('product_cls', self.product_cls.text())
        self.justsaveone(f"data{product_cls}", data)
        self.python_process = subprocess.Popen([_PYTHON_EXE, "-u", _MAKE_STD_SCRIPT], cwd=os.path.join(_REPO_ROOT))
        print(data)

    def _open_cls_config(self):
        """弹出类别配置窗口，关闭后刷新型号展示。"""
        if self._cls_config_window is None or not self._cls_config_window.isVisible():
            self._cls_config_window = ClsConfigWindow(self)
            self._cls_config_window.finished.connect(self._on_cls_config_closed)
        self._cls_config_window.show()
        self._cls_config_window.raise_()
        self._cls_config_window.activateWindow()

    def _on_cls_config_closed(self):
        """类别配置关闭后刷新当前型号展示。"""
        try:
            with open(os.path.join(_REPO_ROOT, 'config', 'rptcfg.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            self.product_cls.setText(str(cfg.get('product_cls', '')))
            self.create_table()
        except Exception as e:
            print(f'刷新型号展示失败: {e}')

    def initUI1(self):
        self.scrollArea01_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(4, 6, 14, 6)
        self.scroll_layout.setSpacing(10)

        # 初始化四个输入框
        # 从 YAML 文件加载数据并初始化输入框
        self.load_class_labels(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"))



        # 设置滚动区域
        self.scrollArea01_widget.setLayout(self.scroll_layout)
        self.scrollArea01.setWidget(self.scrollArea01_widget)
        self.scrollArea01.setWidgetResizable(True)

        # 添加按钮
        self.pushButton_add.clicked.connect(self.add_input_field)

        # 保存按钮
        self.cls_save.clicked.connect(self.save_data)


    def load_class_labels(self, yaml_file):
        """从 YAML 文件加载 class_labels 并初始化输入框"""
        try:
            with open(yaml_file, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                class_labels = data.get("class_labels", {})

                for key, value in class_labels.items():
                    self.add_input_field(key, value)
        except FileNotFoundError:
            print(f"YAML 文件 {yaml_file} 未找到！")
        except yaml.YAMLError as e:
            print(f"解析 YAML 文件时出错: {e}")

    def add_input_field(self, class_id=None, label_text=""):
        """动态添加输入框"""
        if class_id is False:  # 如果未指定 class_id，自动分配一个
            class_id = self.class_counter

        field_layout = QHBoxLayout()
        field_layout.setSpacing(10)
        field_layout.setContentsMargins(0, 2, 0, 2)
        label = QLabel(f"类别 {class_id}:")
        label.setMinimumWidth(76)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        input_field = QLineEdit()
        input_field.setPlaceholderText("请输入类别名称")
        input_field.setText(label_text)  # 初始化输入框内容
        input_field.setMinimumHeight(28)
        remove_button = QPushButton("删除")
        remove_button.setMinimumWidth(52)
        remove_button.setStyleSheet(
            "QPushButton{padding:4px 10px;min-width:48px;}"
        )
        remove_button.clicked.connect(lambda: self.remove_input_field(field_layout))

        field_layout.addWidget(label, 0)
        field_layout.addWidget(input_field, 1)
        field_layout.addWidget(remove_button, 0)

        self.scroll_layout.addLayout(field_layout)

        # 将输入框和布局记录到 self.input_fields 中
        self.input_fields.append((field_layout, input_field))

        self.class_counter += 1

    def add_input_field01(self, class_id=None, label_text=""):
        """动态添加输入框"""
        if class_id is False:  # 如果未指定 class_id，自动分配一个

            class_id = self.class_counter

        field_layout = QHBoxLayout()
        field_layout.setSpacing(10)
        field_layout.setContentsMargins(0, 2, 0, 2)
        label = QLabel(f"类别 {class_id}:")
        label.setMinimumWidth(76)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        input_field = QLineEdit()
        input_field.setPlaceholderText("请输入类别名称")
        input_field.setText(label_text)  # 初始化输入框内容
        input_field.setMinimumHeight(28)
        remove_button = QPushButton("删除")
        remove_button.setMinimumWidth(52)
        remove_button.setStyleSheet(
            "QPushButton{padding:4px 10px;min-width:48px;}"
        )
        remove_button.clicked.connect(lambda: self.remove_input_field(field_layout))

        field_layout.addWidget(label, 0)
        field_layout.addWidget(input_field, 1)
        field_layout.addWidget(remove_button, 0)

        self.scroll_layout.addLayout(field_layout)
        self.class_counter += 1

    def renumber_fields(self):
        """重新计算并刷新所有类别编号"""
        for index, (layout, input_field) in enumerate(self.input_fields):
            # 新的编号是 列表索引 + 1
            new_id = index + 1

            # 获取 layout 中的第一个控件（即 QLabel "类别 x:"）
            # 注意：itemAt(0) 是 label, itemAt(1) 是 input, itemAt(2) 是 button
            label_item = layout.itemAt(0)
            if label_item:
                label_widget = label_item.widget()
                if isinstance(label_widget, QLabel):
                    label_widget.setText(f"类别 {new_id}:")

        # 更新计数器，确保下一次“添加”时，编号是接着当前的最后一位
        self.class_counter = len(self.input_fields) + 1

    def remove_input_field(self, layout_to_remove):
        """删除指定的输入框"""
        for i, (layout, input_field) in enumerate(self.input_fields):
            if layout == layout_to_remove:
                # 1. 从列表中移除记录
                self.input_fields.pop(i)

                # 2. 从界面 UI 中移除控件
                self.clear_layout(layout_to_remove)

                # 3. 【关键步骤】触发重排，更新剩余的编号
                self.renumber_fields()

                # 找到并删除后即可退出循环
                return

    def save_yaml1(self):
        """保存数据到 YAML 文件"""
        class_labels = {}
        for i in range(self.scroll_layout.count()):
            layout = self.scroll_layout.itemAt(i)
            if isinstance(layout, QHBoxLayout):
                label_widget = layout.itemAt(0).widget()  # 获取标签
                input_widget = layout.itemAt(1).widget()  # 获取输入框

                if isinstance(label_widget, QLabel) and isinstance(input_widget, QLineEdit):
                    class_id = int(label_widget.text().split(" ")[1])  # 提取 class_id
                    label_text = input_widget.text()  # 获取用户输入的类别名称

                    class_labels[class_id] = label_text

    @staticmethod
    def clear_layout(layout):
        """清除布局中的所有小部件"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def save_data(self):
        """将用户输入的数据保存为字典格式"""
        self.class_labels = {}
        for idx, (_, input_field) in enumerate(self.input_fields, start=1):
            text = input_field.text().strip()
            if text:  # 忽略空值
                self.class_labels[idx] = text

        if not self.class_labels:
            QMessageBox.warning(self, "警告", "没有输入有效的类别名称！")
            return
        self.config_data["class_labels"] = self.class_labels
        self.class_list = list(self.class_labels.keys())
        self.config_data["class_list"] = self.class_list
        print(self.class_list)
        # 将数据保存到 rpt01.yaml 文件
        self.justsaveone("class_labels", self.class_labels)
        self.justsaveone("class_list", self.class_list)

        self.initUI2()
        self.initUI3()

    def justsaveone(self, name, data):
        try:
            # 读取原有数据
            try:
                with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "r", encoding="utf-8") as file:
                    existing_data = yaml.safe_load(file) or {}
            except FileNotFoundError:
                existing_data = {}

            # 更新数据
            existing_data[name] = data

            # 写入更新后的数据
            with open(os.path.join(_REPO_ROOT, "config", "rptcfg.yaml"), "w", encoding="utf-8") as file:
                yaml.dump(existing_data, file, allow_unicode=True)

            #QMessageBox.information(self, "保存成功", "数据已成功保存到 rptcfg.yaml 文件！")
            print(name, data)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存数据时出错:\n{e}")
            print(f"保存数据时出错: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReportWindow()
    window.show()
    sys.exit(app.exec_())