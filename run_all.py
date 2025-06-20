# run_all.py

import os
import sys
import subprocess
import re
import time
import webbrowser
import shutil
import json
import torch

def main():
    # —— 0. 可调参数 ——  
    DATASET_ID   = "99"                         # 只写“99”
    DATASET_ID_3 = DATASET_ID.zfill(3)          # 自动补成“099”
    PLAN_CONFIG  = "3d_fullres"                 # U-Net 配置
    FOLD         = "0"                          # fold 序号
    NUM_PROC     = "8"                          # 并行进程数
    EPOCHS       = "5"                          # 训练轮数（字符串）
    NAME         = "Heart"                      # 任务名称
    EXAMPLE_RAW  = "la_003_0000.nii.gz"         # Notebook 示例用文件名

    # —— 1. 环境变量集中设置 ——  
    os.environ.update({
        'NNUNET_DATASET_ID':   DATASET_ID,
        'NNUNET_PLAN_CONFIG':  PLAN_CONFIG,
        'NNUNET_FOLD':         FOLD,
        'NNUNET_NUM_PROC':     NUM_PROC,
        'NNUNET_EPOCH':        EPOCHS,
        'NaME':                NAME,
        'example_raw':         EXAMPLE_RAW,
        # —— 请改为你本地的绝对路径 ——  
        'MSD_raw':             r"C:\Users\chen1myn\Documents\GitHub\nnUNet\DATASET\nnUNet_raw\Task02_Heart",
        'MSD_preprocessed_id': DATASET_ID,
        'nnUNet_raw':          r"C:\Users\chen1myn\Documents\GitHub\nnUNet\DATASET\nnUNet_raw",
        'nnUNet_preprocessed': r"C:\Users\chen1myn\Documents\GitHub\nnUNet\DATASET\nnUNet_preprocessed",
        'nnUNet_results':      r"C:\Users\chen1myn\Documents\GitHub\nnUNet\DATASET\nnUNet_results",
    })
    print("✅ 环境变量已设置")

    # —— 2. 构造路径 ——  
    loc = f"Dataset{DATASET_ID_3}_{NAME}"
    preproc_folder = os.path.join(os.environ['nnUNet_preprocessed'], loc)
    model_folder   = os.path.join(
        os.environ['nnUNet_results'], loc,
        f"nnUNetTrainer__nnUNetPlans__{PLAN_CONFIG}"
    )
    ckpt = os.path.join(model_folder, f"fold_{FOLD}", "checkpoint_best.pth")

    # —— 如果预处理 & 训练都已完成，跳过到可视化 ——  
    if os.path.isdir(preproc_folder) and os.path.isfile(ckpt):
        print("✅ 已检测到预处理目录和 checkpoint，直接展示")
        _write_config()
        _execute_notebook()
        _launch_server_and_open("show_executed.ipynb")
        return

    # —— 3. MSD 转换 ——  
    if not os.path.isdir(preproc_folder):
        print("🔄 开始 MSD 数据转换 …")
        from nnunetv2.dataset_conversion.convert_MSD_dataset import run_main_convert
        run_main_convert()
        print("✅ MSD 数据转换完成")
    else:
        print("✅ 已存在预处理目录，跳过转换")

    # —— 4. 规划 + 预处理 ——  
    print("🔄 开始 规划与预处理 …")
    sys.argv = [
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity",
        "-np", NUM_PROC
    ]
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
    plan_and_preprocess_entry()
    print("✅ 规划与预处理完成")

    # —— 5. 训练 ——  
    if not os.path.isfile(ckpt):
        print("🔄 开始 训练 …")
        sys.argv = ["run_training.py", DATASET_ID, PLAN_CONFIG, FOLD]
        from nnunetv2.run.run_training import run_training_entry
        run_training_entry()
        print("✅ 训练完成，checkpoint 已保存")
    else:
        print("✅ 已存在 checkpoint，跳过训练")

    # —— 6. 推理 ——  
    print("🔄 开始 推理，生成 imagesTs_predlowres …")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(int(FOLD),),
        checkpoint_name="checkpoint_best.pth"
    )

    src = os.path.join(os.environ['nnUNet_raw'], loc, "imagesTr")
    ts  = os.path.join(os.environ['nnUNet_raw'], loc, "imagesTs")
    out = os.path.join(os.environ['nnUNet_raw'], loc, "imagesTs_predlowres")
    os.makedirs(ts,  exist_ok=True)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith("_0000.nii.gz"):
            dst = os.path.join(ts, f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(src, f), dst)

    predictor.predict_from_files(
        ts, out,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    print("✅ 推理完成，结果保存在：", out)

    # —— 7. 写 config + 执行 & 展示 Notebook ——  
    _write_config()
    _execute_notebook()
    _launch_server_and_open("show_executed.ipynb")


def _write_config():
    """
    写入 config.json，供 show.ipynb 读取：
    """
    cfg = {
        "nnUNet_raw": os.environ['nnUNet_raw'],
        "datasetid":  os.environ['NNUNET_DATASET_ID'],
        "NAME":       os.environ['NaME']
    }
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("✅ 已生成 config.json")


def _execute_notebook():
    """
    预先执行 show.ipynb 并输出到 show_executed.ipynb
    """
    print("🚀 正在用 nbconvert 运行 show.ipynb …")
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", "show.ipynb",
        "--output", "show_executed.ipynb"
    ], check=True)
    print("✅ Notebook 已执行完毕，输出 show_executed.ipynb")


def _launch_server_and_open(nb_file: str):
    """
    启动 notebook 服务并打开执行好的 nb_file
    """
    cwd = os.getcwd()
    print("🚀 启动 Jupyter Server（--no-browser）…")
    proc = subprocess.Popen(
        ["jupyter", "notebook", "--no-browser", "--notebook-dir", cwd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    base_url = None
    token    = None
    # 从日志里抓 token
    for line in proc.stdout:
        print(line, end="")
        m = re.search(r"(http://[^/\s]+:\d+)(?:/\S*)?\?token=(\w+)", line)
        if m:
            base_url = m.group(1)
            token    = m.group(2)
            break

    if not base_url or not token:
        raise RuntimeError("❌ 无法解析 Jupyter token URL")

    url = f"{base_url}/notebooks/{nb_file}?token={token}"
    print(f"\n🚀 捕获到并打开：\n    {url}\n")
    webbrowser.open(url)
    print("（完成后请在终端按 Ctrl+C 停止服务器）")


if __name__ == "__main__":
    main()
