'''
----------------------------- 代码用途 ---------------------------
作者：小房总
该代码在顾问“华子哥”的源代码基础上，增加了批量检测，以及sharpe值检验的功能，
目前已经能够满足User阶段的自相关检测，批量检查因子能否提交，节省大量时间。
----------------------------- 使用说明 ---------------------------
你需要做的只有两点
1. 同文件夹下创建名为brain_credentials.txt的文件，
里面的格式为：["账号", "密码"]
2. 将ALPHA_LIST里面的值替换成你需要检测的id
3. 终端里运行python3 self-corr-version_1.1.0.py 即可
-----------------------------------------------------------------
Updated：Oct 12, 2025
Version 1.1.0版，新增了推荐提升alpha的功能
推荐提升的alpha，适当修改一下因子的参数，很容易救活这些“死掉”的因子
-----------------------------------------------------------------
Updated：Oct 13, 2025
Version 1.2.0版，将alpha_list和csv文件提升为全局变量
'''

import requests
import pandas as pd
import logging
import time
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor
import pickle
from collections import defaultdict
import numpy as np
from pathlib import Path
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth

# ---------------- 全局参数 ----------------
CORR_CUTOFF = 0.7         # 相关性阈值：<=0.7必Pass；>0.7触发Sharpe对比
SHARPE_PREMIUM = 1.10     # 被测Sharpe至少需高出“相关peer中最大Sharpe”10%
ALPHA_LIST = ["nppwqQ18","YPPqKLgW","j22RNKME","A11MdY7g"]
CSV_FILE = "测试.csv"

# ---------------- 登录 ----------------
def sign_in(username, password):
    s = requests.Session()
    s.auth = (username, password)
    try:
        response = s.post('https://api.worldquantbrain.com/authentication')
        response.raise_for_status()
        logging.info("Successfully signed in")
        return s
    except requests.exceptions.RequestException as e:
        logging.error(f"Login failed: {e}")
        return None

# ----------- 从文件读取账号密码登录 -----------
def sign_in_from_file():
    cred_path = expanduser('brain_credentials.txt')
    try:
        with open(cred_path) as f:
            credentials = json.load(f)
        username, password = credentials
        sess = requests.Session()
        sess.auth = HTTPBasicAuth(username, password)
        response = sess.post('https://api.worldquantbrain.com/authentication')
        response.raise_for_status()
        print("✅ 登录成功（凭证文件）")
        return sess
    except FileNotFoundError:
        print("⚠️ 未找到 brain_credentials.txt 文件，将使用手动登录方式。")
        return None
    except Exception as e:
        print(f"❌ 从文件登录失败：{e}")
        return None

# ---------------- 文件操作 ----------------
def save_obj(obj: object, name: str) -> None:
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name: str) -> object:
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

# ---------------- 请求重试 ----------------
def wait_get(url: str, max_retries: int = 10):
    retries = 0
    while retries < max_retries:
        while True:
            simulation_progress = sess.get(url)
            if simulation_progress.headers.get("Retry-After", 0) == 0:
                break
            time.sleep(float(simulation_progress.headers["Retry-After"]))
        if simulation_progress.status_code < 400:
            break
        else:
            time.sleep(2 ** retries)
            retries += 1
    return simulation_progress

# ---------------- 获取单个 Alpha PnL ----------------
def _get_alpha_pnl(alpha_id: str) -> pd.DataFrame:
    pnl = wait_get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/recordsets/pnl").json()
    df = pd.DataFrame(pnl['records'], columns=[item['name'] for item in pnl['schema']['properties']])
    df = df.rename(columns={'date':'Date', 'pnl':alpha_id})
    df = df[['Date', alpha_id]]
    return df

# ---------------- 批量获取 PnL ----------------
def get_alpha_pnls(alphas: list[dict],
                   alpha_pnls: Optional[pd.DataFrame] = None,
                   alpha_ids: Optional[dict[str, list]] = None) -> Tuple[dict[str, list], pd.DataFrame]:
    if alpha_ids is None:
        alpha_ids = defaultdict(list)
    if alpha_pnls is None:
        alpha_pnls = pd.DataFrame()

    new_alphas = [item for item in alphas if item['id'] not in alpha_pnls.columns]
    if not new_alphas:
        return alpha_ids, alpha_pnls

    for item_alpha in new_alphas:
        alpha_ids[item_alpha['settings']['region']].append(item_alpha['id'])

    fetch_pnl_func = lambda alpha_id: _get_alpha_pnl(alpha_id).set_index('Date')
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_pnl_func, [item['id'] for item in new_alphas])
    alpha_pnls = pd.concat([alpha_pnls] + list(results), axis=1)
    alpha_pnls.sort_index(inplace=True)
    return alpha_ids, alpha_pnls

# ---------------- 获取 OS Alpha 列表 ----------------
def get_os_alphas(limit: int = 100, get_first: bool = False) -> List[Dict]:
    fetched_alphas = []
    offset = 0
    total_alphas = 100
    while len(fetched_alphas) < total_alphas:
        print(f"Fetching alphas from offset {offset} to {offset + limit} ...")
        url = f"https://api.worldquantbrain.com/users/self/alphas?stage=OS&limit={limit}&offset={offset}&order=-dateSubmitted"
        res = wait_get(url).json()
        if offset == 0:
            total_alphas = res['count']
            print(f"🔍 共发现 {total_alphas} 条 OS alpha，准备获取最新 {limit if get_first else total_alphas} 条...")
        alphas = res["results"]
        fetched_alphas.extend(alphas)
        print(f"✅ 本次已获取 {len(fetched_alphas)} 条")

        if len(alphas) < limit or get_first:
            break
        offset += limit
    print(f"✅ OS alpha 列表获取完成，共返回 {len(fetched_alphas)} 条")
    print("正在下载，请耐心等待...")
    return fetched_alphas[:total_alphas]

# ---------------- 获取 Sharpe 值（带简单运行期缓存） ----------------
_sharpe_cache_runtime: Dict[str, float] = {}

def get_alpha_sharpe(alpha_id: str) -> float:
    """从 API 获取单个 alpha 的 Sharpe 值；运行期缓存避免重复IO"""
    if alpha_id in _sharpe_cache_runtime:
        return _sharpe_cache_runtime[alpha_id]
    try:
        data = wait_get(f"https://api.worldquantbrain.com/alphas/{alpha_id}").json()
        checks = data.get("is", {}).get("checks", [])
        match = next((c for c in checks if c.get("name") == "LOW_SHARPE"), None)
        if match and "value" in match:
            val = float(match["value"])
        elif match and "result" in match and isinstance(match["result"], (int, float)):
            val = float(match["result"])
        else:
            val = np.nan
    except Exception as e:
        print(f"⚠️ 获取 {alpha_id} 的 Sharpe 值失败: {e}")
        val = np.nan
    _sharpe_cache_runtime[alpha_id] = val
    return val

# ---------------- 计算单个 Alpha 自相关（返回全序列） ----------------
def calc_self_corr_series(alpha_id: str,
                          os_alpha_rets: pd.DataFrame | None = None,
                          os_alpha_ids: dict[str, str] | None = None,
                          alpha_result: dict | None = None,
                          alpha_pnls: pd.DataFrame | None = None) -> pd.Series:
    if alpha_result is None:
        alpha_result = wait_get(f"https://api.worldquantbrain.com/alphas/{alpha_id}").json()
    if alpha_pnls is not None and len(alpha_pnls) == 0:
        alpha_pnls = None
    if alpha_pnls is None:
        _, alpha_pnls = get_alpha_pnls([alpha_result])
        alpha_pnls = alpha_pnls[alpha_id]
    alpha_rets = alpha_pnls - alpha_pnls.ffill().shift(1)
    alpha_rets = alpha_rets[pd.to_datetime(alpha_rets.index) > pd.to_datetime(alpha_rets.index).max() - pd.DateOffset(years=4)]
    region = alpha_result['settings']['region']
    pool = os_alpha_rets[os_alpha_ids[region]]
    corr_series = pool.corrwith(alpha_rets).sort_values(ascending=False).round(4)
    return corr_series

# ---------------- 下载数据 ----------------
def download_data(flag_increment=True):
    if flag_increment:
        try:
            os_alpha_ids = load_obj(str(cfg.data_path / 'os_alpha_ids'))
            os_alpha_pnls = load_obj(str(cfg.data_path / 'os_alpha_pnls'))
            ppac_alpha_ids = load_obj(str(cfg.data_path / 'ppac_alpha_ids'))
            exist_alpha = [alpha for ids in os_alpha_ids.values() for alpha in ids]
        except Exception:
            os_alpha_ids = None
            os_alpha_pnls = None
            exist_alpha = []
            ppac_alpha_ids = []
    else:
        os_alpha_ids = None
        os_alpha_pnls = None
        exist_alpha = []
        ppac_alpha_ids = []
    alphas = get_os_alphas(limit=100, get_first=False)

    alphas = [item for item in alphas if item['id'] not in exist_alpha]
    ppac_alpha_ids += [item['id'] for item in alphas for item_match in item['classifications'] if item_match['name'] == 'Power Pool Alpha']
    os_alpha_ids, os_alpha_pnls = get_alpha_pnls(alphas, alpha_pnls=os_alpha_pnls, alpha_ids=os_alpha_ids)
    save_obj(os_alpha_ids, str(cfg.data_path / 'os_alpha_ids'))
    save_obj(os_alpha_pnls, str(cfg.data_path / 'os_alpha_pnls'))
    save_obj(ppac_alpha_ids, str(cfg.data_path / 'ppac_alpha_ids'))
    print(f'新下载的alpha数量: {len(alphas)}, 目前总共提交alpha数量: {os_alpha_pnls.shape[1]}')

# ---------------- 加载数据 ----------------
def load_data(tag=None):
    os_alpha_ids = load_obj(str(cfg.data_path / 'os_alpha_ids'))
    os_alpha_pnls = load_obj(str(cfg.data_path / 'os_alpha_pnls'))
    ppac_alpha_ids = load_obj(str(cfg.data_path / 'ppac_alpha_ids'))
    if tag == 'PPAC':
        for item in os_alpha_ids:
            os_alpha_ids[item] = [alpha for alpha in os_alpha_ids[item] if alpha in ppac_alpha_ids]
    elif tag == 'SelfCorr':
        for item in os_alpha_ids:
            os_alpha_ids[item] = [alpha for alpha in os_alpha_ids[item] if alpha not in ppac_alpha_ids]
    exist_alpha = [alpha for ids in os_alpha_ids.values() for alpha in ids]
    os_alpha_pnls = os_alpha_pnls[exist_alpha]
    os_alpha_rets = os_alpha_pnls - os_alpha_pnls.ffill().shift(1)
    os_alpha_rets = os_alpha_rets[pd.to_datetime(os_alpha_rets.index) > pd.to_datetime(os_alpha_rets.index).max() - pd.DateOffset(years=4)]
    return os_alpha_ids, os_alpha_rets

# ---------------- 配置类 ----------------
class cfg:
    username = ""
    password = ""
    data_path = Path('.')

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    sess = sign_in_from_file()
    if sess is None:
        sess = sign_in(cfg.username, cfg.password)

    download_data(flag_increment=True)
    os_alpha_ids, os_alpha_rets = load_data()

    print(f"即将测试的alpha数量为 {len(ALPHA_LIST)} 条")

    results = {}
    for idx, alpha_id in enumerate(ALPHA_LIST, start=1):
        try:
            corr_series = calc_self_corr_series(alpha_id, os_alpha_rets=os_alpha_rets, os_alpha_ids=os_alpha_ids)
            max_corr = float(corr_series.max()) if not corr_series.empty else 0.0
            max_corr_id = str(corr_series.idxmax()) if not corr_series.empty else None
            sharpe_current = get_alpha_sharpe(alpha_id)

            status = "Pass"
            peer_ids_over = corr_series[corr_series > CORR_CUTOFF].index.tolist()
            max_peer_sharpe = np.nan
            if len(peer_ids_over) > 0:
                peer_sharpes = [get_alpha_sharpe(pid) for pid in peer_ids_over if not np.isnan(get_alpha_sharpe(pid))]
                if len(peer_sharpes) == 0:
                    status = "Fail (Peers>0.7 but Sharpe missing)"
                else:
                    max_peer_sharpe = max(peer_sharpes)
                    if np.isnan(sharpe_current) or sharpe_current < SHARPE_PREMIUM * max_peer_sharpe:
                        status = "Fail"
                    else:
                        status = "Pass"
            else:
                status = "Pass"

            results[alpha_id] = {
                "Corr_Max": max_corr,
                "Corr_Max_ID": max_corr_id,
                "Corr_Cutoff": CORR_CUTOFF,
                "Num_Peers_Over_Cutoff": len(peer_ids_over),
                "Sharpe_Current": sharpe_current,
                "Sharpe_Peers_MaxOverCutoff": max_peer_sharpe,
                "Sharpe_Premium": SHARPE_PREMIUM,
                "Result": status
            }

            if len(peer_ids_over) == 0:
                print(f"{idx}. {alpha_id} - {status} | MaxCorr={max_corr:.4f} (<= {CORR_CUTOFF}), Sharpe={sharpe_current:.3f}")
            else:
                print(f"{idx}. {alpha_id} - {status} | MaxCorr={max_corr:.4f} (> {CORR_CUTOFF}), "
                      f"Sharpe={sharpe_current:.3f}, MaxPeerSharpe={max_peer_sharpe if not np.isnan(max_peer_sharpe) else float('nan'):.3f}, "
                      f"PeersOver={len(peer_ids_over)}")

        except Exception as e:
            results[alpha_id] = {"Result": f"Error - {str(e)}"}
            print(f"{idx}. {alpha_id}: Error - {e}")

    # 输出 CSV
    result_df = pd.DataFrame([{"Alpha_ID": k, **v} for k, v in results.items()])
    result_df.to_csv(f"{CSV_FILE}", index=False)

    # 汇总输出
    total = len(results)
    pass_ids = [k for k, v in results.items() if v.get("Result") == "Pass"]
    fail_count = total - len(pass_ids)
    print("\n" + "=" * 80)
    print(f"Result：共执行 {total} 条记录。Pass = {len(pass_ids)}，Fail = {fail_count}")

    if len(pass_ids) > 0:
        print("通过的 Alpha ID：")
        for i in range(0, len(pass_ids), 10):
            group = pass_ids[i:i + 10]
            print("  " + ", ".join(group))
    else:
        print("没有通过的 Alpha。")

    print("=" * 80)

    # ---------------- 推荐提升 Alpha ----------------
    recommend_ids = []
    for k, v in results.items():
        if (
            v.get("Result") == "Fail"
            and v.get("Num_Peers_Over_Cutoff") == 1
            and not np.isnan(v.get("Sharpe_Peers_MaxOverCutoff"))
            and v.get("Sharpe_Peers_MaxOverCutoff") > 0
            and v.get("Sharpe_Current", 0) / v.get("Sharpe_Peers_MaxOverCutoff") > 1.05
        ):
            recommend_ids.append(k)

    print("\n" + "=" * 80)
    print(f"推荐提升Alpha：共{len(recommend_ids)}条记录")
    if recommend_ids:
        print("Alpha ID：")
        for i in range(0, len(recommend_ids), 10):
            group = recommend_ids[i:i + 10]
            print("  " + ", ".join(group))
    else:
        print("暂无符合条件的推荐Alpha。")
    print("=" * 80)

    print(f"\n检测完成，结果已保存到 {CSV_FILE} ✅")
