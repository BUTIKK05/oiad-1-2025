import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ------------------ Путь к данным ------------------
DATA_PATH = "C:/Users/BUTIK/Documents/labsOIAD/datasets/teen_phone_addiction_dataset.csv"

# ------------------ Создание папки для графиков ------------------
plots_dir = "./plots"
os.makedirs(plots_dir, exist_ok=True)

# ------------------ Чтение данных ------------------
print(f"Чтение {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ------------------ Функции ------------------
def mode_continuous(x):
    x = np.asarray(x)[~np.isnan(x)]
    if len(x)==0: return np.nan
    try:
        bw = 3.5 * x.std(ddof=1) / (len(x) ** (1/3))
        bins = max(int(np.ceil((x.max()-x.min())/bw)), 10)
    except Exception:
        bins = 10
    counts, edges = np.histogram(x, bins=bins)
    idx = np.argmax(counts)
    return (edges[idx] + edges[idx+1]) / 2.0

def descriptive_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n==0:
        return {}
    mean = x.mean()
    var = x.var(ddof=1) if n>1 else np.nan
    med = np.median(x)
    q25, q50, q75 = np.quantile(x, [0.25,0.5,0.75])
    iqr = q75 - q25
    mode = mode_continuous(x)
    skew = stats.skew(x, bias=False) if n>2 else np.nan
    kurt = stats.kurtosis(x, fisher=True, bias=False) if n>3 else np.nan
    return {"n":n, "mean":mean, "variance":var, "mode":mode, "median":med,
            "q25":q25, "q50":q50, "q75":q75, "iqr":iqr, "skewness":skew, "excess_kurtosis":kurt}

def ecdf(x):
    xs = np.sort(x[~np.isnan(x)])
    ys = np.arange(1, len(xs)+1)/len(xs)
    return xs, ys

def chi_square_normality_manual(x, bins=None):
    x = np.asarray(x)[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return {"chi2":np.nan, "df":np.nan, "p_value":np.nan, "note":"n<10"}
    mu = x.mean(); sigma = x.std(ddof=1)
    if sigma <= 0:
        return {"chi2":np.nan, "df":np.nan, "p_value":np.nan, "note":"sigma<=0"}
    if bins is None:
        bins = max(6, int(np.ceil(np.log2(n) + 1)))
    counts, edges = np.histogram(x, bins=bins)
    cdf = stats.norm(loc=mu, scale=sigma).cdf
    probs = np.diff(cdf(edges))
    expected = probs * n
    obs = counts.astype(float).tolist()
    exp = expected.tolist()
    i = 0
    while i < len(exp):
        if exp[i] < 5:
            if i == 0:
                exp[i+1] += exp[i]
                obs[i+1] += obs[i]
                del exp[i]; del obs[i]
            else:
                exp[i-1] += exp[i]
                obs[i-1] += obs[i]
                del exp[i]; del obs[i]
                i -= 1
            continue
        i += 1
    k = len(exp)
    if k <= 3:
        return {"chi2":np.nan, "df":np.nan, "p_value":np.nan, "note":"too few bins after merging"}
    chi2 = np.sum((np.array(obs)-np.array(exp))**2 / np.array(exp))
    df = k - 1 - 2
    p_value = 1 - stats.chi2.cdf(chi2, df) if df>0 else np.nan
    return {"chi2":chi2, "df":df, "p_value":p_value, "bins_used":k}

def jarque_bera_manual(x):
    x = np.asarray(x)[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"JB":np.nan, "p_value":np.nan}
    s = stats.skew(x, bias=False); k = stats.kurtosis(x, fisher=True, bias=False)
    JB = n/6.0 * (s**2 + (k**2)/4.0)
    p = 1 - stats.chi2.cdf(JB, df=2)
    return {"JB":JB, "p_value":p, "skewness":s, "excess_kurtosis":k}

def winsorize_iqr(x, k=1.5):
    arr = np.asarray(x, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum()==0: return arr
    a = arr[mask]
    q25,q75 = np.quantile(a,[0.25,0.75]); iqr = q75-q25
    low = q25 - k*iqr; high = q75 + k*iqr
    res = arr.copy(); res[mask] = np.clip(a, low, high)
    return res

# ------------------ Пункт I: Описательные ------------------
target = "Daily_Usage_Hours"
orig = df[target].values
stats_orig = descriptive_stats(orig)
print("I. Описательные характеристики (исходные данные):")
print(stats_orig)

# --- Графики ---
plt.hist(orig, bins='auto', density=True)
plt.title("Гистограмма (исходные)")
plt.xlabel(target)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "hist_original.png"))
plt.close()

xs, ys = ecdf(orig)
plt.step(xs, ys, where='post')
plt.title("ECDF (исходные)")
plt.xlabel(target)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ecdf_original.png"))
plt.close()

n = len(orig)
mu = orig.mean(); sigma = orig.std(ddof=1)
probs = (np.arange(1,n+1)-0.5)/n
theo = stats.norm.ppf(probs, loc=mu, scale=sigma)
plt.scatter(theo, np.sort(orig), s=10)
mn=min(theo.min(), orig.min()); mx=max(theo.max(), orig.max())
plt.plot([mn,mx],[mn,mx], ls='--')
plt.title("Q-Q plot (исходные)")
plt.xlabel("Теор. квант")
plt.ylabel("Набл. квант")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "qq_original.png"))
plt.close()

# ------------------ Пункт II: Проверка нормальности ------------------
chi_o = chi_square_normality_manual(orig)
jb_o = jarque_bera_manual(orig)
print("\nII. Тесты нормальности (исходные данные):")
print("Chi-square:", chi_o)
print("Jarque-Bera:", jb_o)

# ------------------ Пункт III: Обработка данных ------------------
wins = winsorize_iqr(orig, k=1.5)
minv = wins.min()
shift = 0
if minv <= 0:
    shift = abs(minv) + 1.0
logw = np.log1p(wins + shift)
zlogw = (logw - np.mean(logw)) / logw.std(ddof=1)

# --- Статистика обработанных данных ---
stats_wins = descriptive_stats(wins)
stats_logw = descriptive_stats(logw)
stats_zlogw = descriptive_stats(zlogw)
print("\nIII. Описательные характеристики (обработанные данные):")
print("Winsorized:", stats_wins)
print("Log(Wins):", stats_logw)
print("Z(Log):", stats_zlogw)

# --- Графики обработанных ---
for arr, name in zip([wins, logw, zlogw], ["winsorized","logw","zlogw"]):
    plt.hist(arr, bins='auto', density=True)
    plt.title(f"Гистограмма ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"hist_{name}.png"))
    plt.close()
    xs, ys = ecdf(arr)
    plt.step(xs, ys, where='post')
    plt.title(f"ECDF ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"ecdf_{name}.png"))
    plt.close()
    # Q-Q plot
    plt.scatter(stats.norm.ppf(probs, loc=arr.mean(), scale=arr.std(ddof=1)), np.sort(arr), s=10)
    mn=min(arr.min(), theo.min()); mx=max(arr.max(), theo.max())
    plt.plot([mn,mx],[mn,mx], ls='--')
    plt.title(f"Q-Q plot ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"qq_{name}.png"))
    plt.close()

# ------------------ Пункт IV: Группировка по School_Grade ------------------
group_stats = df.groupby("School_Grade")[target].agg(['count','mean','var']).rename(columns={'var':'variance'})
print("\nIV. По группам School_Grade:")
print(group_stats)

plt.figure()
for g, sub in df.groupby("School_Grade"):
    vals = sub[target].values
    plt.hist(vals, bins='auto', alpha=0.5, density=True, label=str(g))
plt.legend(); plt.title("Гистограммы по School_Grade")
plt.xlabel(target)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "hist_by_group.png"))
plt.close()

print("\nВсе графики сохранены в папке:", os.path.abspath(plots_dir))
print("Скрипт завершил работу.")
