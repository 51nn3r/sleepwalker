# %% [markdown]
# # Sleepwalker: Dreamer с действиями-запросами — актор, модель мира, VM, RM, критик, Shapley (рыцари и лжецы, рекурсия)
#
# Описание модели — `docs/model_wm.md`, судья — `docs/judge.md`. Что делает этот ноутбук:
# 1. **Датасет.** Базовая LLM решает все задачи обучения по полному тексту состояния, без обучения, по 2 попытки на
#    задачу; судья (сильная модель через OpenRouter) сравнивает пары попыток — на них VM учится попарно (ваше).
# 2. **Модель мира** (энкодер, декодер, RSSM — денойзеры на предобученной BERT-подобной модели: ModernBERT-base,
#    окно 512 токенов; в арифметике — distilroberta-base). Порядок (ваше): сначала **VM**, её игра Shapley даёт метки
#    **RM**; затем вся модель мира учится разом, вместе с RM (как голова награды в DreamerV3). Окно сначала учится
#    копировать сжатый текст состояния (предл.), затем — обычные потери модели мира.
# 3. **π0.** Актор учится действовать только по окну — на своих удачных попытках (учителя нет). Если π0 по окну намного
#    хуже стартовой политики по тексту — в логе предупреждение.
# 4. **Итерации:** настоящие прогоны → VM и метки RM (Shapley по VM) → модель мира вместе с RM → воображение: критик и актор
#    (ваша формула −Σ φ·log π + β·KL, φ — Shapley строк шага по критику).
# 5. **Сравнение:** прямой ответ базовой модели; стартовая политика; π0; модель с моделью мира; GRPO по настоящим
#    прогонам без модели мира (тот же бюджет исполнений).
#
# Задачи (Config.task): arith — вложенная арифметика, например `Compute ((23 + 7) * (12 - 5)) - (8 * 9)`;
# kk — «рыцари и лжецы» (официальный набор Knights & Knaves, на котором уже оценивали модели в публикациях);
# gsm8k — школьные текстовые задачи. Актор отправляет запросы в LLM (та же модель без адаптера) или на следующий
# уровень (та же модель, рекурсия), затем пишет ANSWER. Награда R = 1 только наверху, если ответ верный.
#
# Из этого файла собираются два ноутбука: sleepwalker.ipynb (быстрый: официальный Knights & Knaves, Qwen-1.5B) и
# sleepwalker_pro.ipynb (тот же K&K и Qwen-1.5B, больше данных и итераций). Отличаются только строкой PRESET.
#
# Запуск: Runtime → Change runtime type → A100 (или G4), затем Run all. На Google Drive нужно ≈30 ГБ (pro; fast ≈15):
# ≈8 ГБ — сами файлы прогона, остальное — корзина: заменённые копии весов Drive кладёт в корзину (каждая итерация
# ≈1,9 ГБ на оба плеча) — при долгих прогонах очищайте её по ходу, иначе место кончится.
# Ключ OpenRouter для судьи — в ячейке ниже (OPENROUTER_API_KEY = "…"; ноутбук с ключом не выкладывайте) или в Colab
# Secrets под именем OPENROUTER_API_KEY.
# Всё пишется на Google Drive в MyDrive/sleepwalker/<имя прогона>/: logs/ — всё лёгкое для анализа (журнал, настройки,
# итоги, траектории, ответы судьи; её можно скачать целиком), weights/ — веса. После обрыва Run all продолжает.

# %% [markdown]
# ## 0. Установка и настройки

# %%
# !pip install -q -U transformers peft accelerate datasets
# !pip uninstall -y -q torchao   # старый torchao в Colab ломает создание LoRA в новом peft; он не нужен

import ast
import contextlib
import gc
import gzip
import zlib
import copy
import itertools
import json
import math
import os
import random
import re
import shutil
import socket
import sys
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, asdict, replace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # до импорта torch
import sys
if "--gpu" in sys.argv:                                 # на сервере: python3 sleepwalker.py --gpu 1 — какую карту занять
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]   # (тоже до импорта torch)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DRY_RUN = os.environ.get("DRY_RUN") == "1" or "--dry" in sys.argv   # локальная проверка без GPU: модели-заглушки
OPENROUTER_API_KEY = ""   # ключ OpenRouter для судьи — прямо в ноутбуке (ваше). В исходнике и репозитории пусто: ключ
#                           подставляется при сборке ноутбука (OPENROUTER_API_KEY=... python3 build_notebook.py …)
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
if "--status" not in sys.argv and "--stop" not in sys.argv:
    print("device", DEVICE, DTYPE, "| dry_run", DRY_RUN, "| torch", torch.__version__)
if not DRY_RUN and "--status" not in sys.argv and "--stop" not in sys.argv:   # работают и без пакетов моделей
    import transformers
    import peft
    print("transformers", transformers.__version__, "| peft", peft.__version__)


def from_pretrained(cls, path):
    """Новые transformers принимают dtype=, старые — torch_dtype=."""
    try:
        return cls.from_pretrained(path, dtype=DTYPE)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=DTYPE)


# %% [markdown]
# ## 1. Настройки эксперимента
# Значения по умолчанию рассчитаны на A100 80 ГБ: полный прогон ≈ 3–4 ч (оценка, не замер).
# Пометка «(предл.)» — мой выбор там, где в описании модели решения ещё нет.

# %%
@dataclass
class Config:
    # LLM
    actor_model: str = "Qwen/Qwen2.5-1.5B-Instruct"   # актор; без адаптера он же исполняет запросы «LLM»
    vm_model: str = "answerdotai/ModernBERT-base"   # BERT-подобная; декодер (например, Qwen) тоже можно — с LoRA
    actor_lora_r: int = 64
    vm_lora_r: int = 32
    # задача: вложенная арифметика
    depth: int = 3                  # глубина дерева выражения (30% задач — на 1 меньше)
    leaf_max: int = 40
    max_level: int = 16             # глубина рекурсии: 0 — верхний уровень; SUB допустим, пока уровень < max_level
    sub_top: int = 16               # своих подзадач у задачи уровня 0; на каждом уровне ниже — вдвое меньше (ваше),
    sub_min: int = 2                # но не меньше 2 (ваше)
    sub_total: int = 16             # всего подзадач на одну задачу верхнего уровня (общий счётчик; иначе дерево
    #                                 из разборов случаев удваивается до 2^16). Кончился счётчик — SUB не предлагается
    max_steps: int = 4              # шагов актора на уровне
    max_reqs: int = 4               # запросов в шаге (SCAR: 2^k коалиций)
    task: str = "arith"             # arith | kk (рыцари и лжецы) | gsm8k
    kk_source: str = "gen"          # kk: hf — официальный набор K&K; gen — наш генератор в том же формате
    kk_people: tuple = (3, 7)       # kk: жителей в задачах для обучения
    kk_eval_people: tuple = (2, 8)  # kk: жителей в отложенных задачах (как в публикациях)
    kk_depth: int = 1               # генератор: 0 — только «X — рыцарь/лжец»; 1 — ещё и/или/если/тогда и только тогда
    max_answer_chars: int = 40      # длина ответа и результата запроса в состоянии
    # объёмы данных
    dataset_tasks: int = 384        # задач в датасете (0 — все задачи обучения); на каждую — dataset_attempts попыток
    n_iter_tasks: int = 128         # задач на итерацию (настоящее исполнение)
    iter_attempts: int = 2          # попыток на задачу на итерации (ваше): пары попыток → судья → VM попарно
    n_eval: int = 300               # отложенные задачи для оценки
    iterations: int = 15
    eval_every: int = 5
    replay_iters: int = 5           # сколько последних итераций держать в буфере (плюс часть датасета)
    # модель мира (ваше: окно 128 токенов, денойзеры, STE, потеря RSSM идёт в энкодер)
    window: int = 128
    max_state_tokens: int = 320
    max_action_tokens: int = 96     # действие во входе модели мира (её токены)
    gen_action_tokens: int = 96     # сколько токенов актор может написать за ход
    vocab_cap: int = 6000           # компактный словарь модели мира (предл.)
    wm_dim: int = 384
    wm_layers: int = 4
    wm_heads: int = 6
    wm_backbone: str = "distilroberta-base"   # предобученная BERT-подобная модель для энкодера, декодера и RSSM
    wm_grad_ckpt: bool = False      # контрольные точки градиента в модели мира (длинные входы)
    wm_lr: float = 1e-4             # дообучение предобученной модели
    gate_copy_acc: float = 0.5      # ранняя остановка: доля верно скопированного содержимого окна в прогоне (предл.)
    gate_copy_acc_check: float = 0.3   # то же на быстрой проверке (там меньше шагов)
    wm_copy_steps: int = 1500       # фаза 1 (предл.): энкодер учится писать в окно сжатый текст состояния
    wm_boot_steps: int = 1500       # фаза 2 на датасете: потери модели мира (декодер + KL)
    rep_warmup: int = 500           # шагов, за которые вес KL(энкодер ‖ RSSM) растёт от 0 (против схлопывания окна)
    copy_anchor: float = 0.05       # малый вес копирования и после фазы 1 (предл.: окно остаётся читаемым)
    wm_iter_steps: int = 400        # шагов модели мира на итерацию
    cf_per_step: int = 2            # контрфактов на настоящий шаг для RSSM (предл.; 0 — выключить)
    wm_batch: int = 64
    topk_ste: int = 8               # STE: вперёд — выбранный токен, назад — смесь top-k
    dec_samples: int = 3            # декодер: несколько зашумлённых версий s (ваше), у каждой — случайная доля шума
    dec_noise_lo: float = 0.15
    dec_noise_hi: float = 0.95
    rssm_samples: int = 2           # RSSM: разных зашумлённых выборок с каждой пары окон (ваше)
    rssm_start_noise: float = 0.5   # предсказание: старт — прошлое окно с такой долей [MASK] (как у x в обучении)
    dyn_scale: float = 0.5          # как в DreamerV3
    rep_scale: float = 0.1
    free_nats: float = 0.05         # на позицию окна (предл.)
    unimix: float = 0.01
    denoise_steps: int = 4          # шагов очистки при предсказании окна
    # RM, критик, актор в воображении (как в DreamerV3)
    head_dim: int = 256
    head_layers: int = 2
    rm_loss_scale: float = 1.0      # вес потери RM внутри модели мира (метки приведены к единичному разбросу)
    critic_lr: float = 3e-4
    horizon: int = 3
    imag_batch: int = 96
    imag_rounds: int = 4
    gamma: float = 0.97
    lam_ret: float = 0.95
    repval_scale: float = 0.3
    ema_decay: float = 0.98         # на каждый шаг критика
    critic_steps: int = 32          # шагов критика за раунд воображения (цели раунда неизменны)
    critic_batch: int = 64
    actor_batch: int = 32           # мини-пачка актора (и в GRPO)
    actor_credit: str = "user_phi"  # user_phi — ваше: −Σ φ·log π + β·KL; adv_plus_centered — A_t + λ·(φ − среднее)
    scar_lambda: float = 0.5        # для adv_plus_centered (предл.)
    kl_beta: float = 0.05
    actor_lr: float = 5e-5
    temperature: float = 1.0
    dataset_temperature: float = 0.7   # датасет: ниже, чем в RL (1.0), — больше удач для VM и π0 (предл.)
    # VM
    vm_lr: float = 5e-5             # VM дообучается целиком
    vm_epochs: int = 1
    sub_weight: float = 0.3         # вес примеров-подзадач в обучении VM; метка — исход задачи наверху (предл.)
    vm_pos_weight_max: float = 8.0  # удачи редки (~2 %): их вес в обучении VM — до ×8 (предл.); выход — снова P успеха
    vm_iter_examples: int = 2000    # VM на итерации: все удачи буфера + случайные неудачи (предл., ради времени)
    # VM попарно (ваше) и судья пар (предл.). Ключ OpenRouter — не здесь: Config целиком пишется в log.txt
    vm_pairs: str = "judge"         # judge — пары по R и по судье; R — только по R (без учителя); none — поточечно
    vm_point_weight: float = 0.3    # вес поточечной части рядом с попарной (шкала и входы подзадач)
    vm_pair_epochs: int = 2         # проходов по парам на предобучении
    vm_iter_pairs: int = 800        # пар из датасета на итерации (повторяются вместе с парами новых эпизодов)
    online_pair_iters: int = 5      # пары скольких последних итераций VM повторяет (как буфер)
    judge_model: str = "openai/gpt-oss-120b"
    judge_extra: str = '{"reasoning": {"effort": "low"}}'   # доп. поля запроса (JSON)
    judge_fallback_model: str = "openai/gpt-oss-120b"     # если основная не отвечает (лимит частоты и т. п.); "" — нет
    judge_fallback_extra: str = '{"reasoning": {"effort": "low"}}'
    judge_tries: int = 3            # попыток к основной модели на запрос (паузы 1 и 2 с), потом — запасная
    judge_max_tokens: int = 1200    # с рассуждением модели; короче — дешевле, но может не дойти до вердикта
    judge_workers: int = 16         # параллельных запросов
    judge_budget_usd: float = 3.0   # предел расхода на судью за весь датасет (дальше пары — только по R)
    judge_cost_guess: float = 0.0004  # $ на запрос до первых ответов (для предела)
    judge_both_frac: float = 0.1    # доля пар с одинаковым исходом, которые судятся в обоих порядках (смещение)
    judge_check_frac: float = 1.0   # доля пар с разным исходом, которые тоже идут судье (проверка по R; их мало)
    vm_refresh: float = 0.25        # доля старых кадров, у которых метки Shapley пересчитываются (новые — всегда)
    vm_max_tokens: int = 384
    vm_batch: int = 16              # текстов в пачке обучения VM (попарно — vm_batch/2 пар)
    vm_grad_ckpt: bool = True       # контрольные точки градиента в VM (длинные входы, мало памяти)
    # датасет: попытки базовой модели по полному тексту (ваше: все задачи, по 2 попытки — пары для VM)
    dataset_attempts: int = 2       # попыток на задачу
    dataset_parts: int = 4          # датасет собирается частями: обрыв теряет не больше одной части
    wm_attempts: int = 0            # попыток датасета для модели мира, RM и буфера (0 — все; предл.: шагов обучения
    #                                 столько же, а игра VM и переходы по всему датасету — лишние часы)
    # π0: обучение по своим удачным попыткам
    bc_max_steps: int = 6000        # не больше стольких шагов удачных попыток датасета
    bc_epochs: int = 2
    bc_lr: float = 1e-4
    # базовая линия GRPO (без модели мира)
    grpo_group: int = 4             # попыток на задачу в GRPO; задач — n_iter_tasks·iter_attempts/grpo_group
    # эксперимент
    arms: tuple = ("full", "grpo_text")
    seeds: tuple = (0,)
    gen_batch: int = 128            # подсказок в одном generate (не больше)
    gen_tokens: int = 96000         # и токенов (подсказка + новые, с заполнением); не влезло в память — пачка делится
    lp_batch: int = 8
    dry_run: bool = False

    @staticmethod
    def dry(**kw):
        """Заглушки вместо LLM, крошечные модели: проверка кода на CPU за пару минут."""
        base = dict(depth=2, max_level=2, sub_top=4, sub_total=6, dataset_tasks=16, n_iter_tasks=8, n_eval=8,
                    iterations=2, iter_attempts=2,
                    eval_every=1, window=16,
                    max_state_tokens=96, max_action_tokens=48, gen_action_tokens=48, wm_dim=32, wm_layers=1,
                    wm_heads=2, wm_copy_steps=15,
                    wm_boot_steps=15, rep_warmup=10,
                    wm_batch=8, head_dim=32, head_layers=1, horizon=2, imag_batch=6, imag_rounds=1, bc_epochs=1,
                    critic_steps=4, actor_batch=4, wm_iter_steps=10,
                    grpo_group=2, gen_batch=8, lp_batch=4, dataset_parts=3, wm_attempts=24, bc_max_steps=200,
                    dry_run=True)
        if kw.get("task") == "kk":
            base.update(kk_people=(3, 4), kk_eval_people=(3, 4), max_answer_chars=300, max_state_tokens=160,
                        max_action_tokens=160, gen_action_tokens=160)
        base.update(kw)
        return Config(**base)

    @staticmethod
    def fast(**kw):
        """Быстрый эксперимент — проверить всё целиком: официальный K&K на 2–8 жителях, как в публикациях; актор
        Qwen2.5-1.5B; VM и модель мира — ModernBERT-base, окно 512 токенов (головоломка целиком влезает).
        Предобучение (один раз, ≈3–4 ч): датасет 1000 задач × 2 попытки (судья — в фоне), VM попарно, модель мира
        1000 + 1000 шагов на 1000 попытках, π0 обоих плеч. Эксперимент (≈2,5–3,5 ч): 8 итераций по 32 задачи.
        Времена — оценки по темпу прошлых прогонов, не замер."""
        base = dict(task="kk", kk_source="hf", actor_model="Qwen/Qwen2.5-1.5B-Instruct",
                    max_steps=5, max_answer_chars=300,
                    kk_people=(2, 8), kk_eval_people=(2, 8), dataset_tasks=1000, dataset_attempts=2, dataset_parts=8,
                    wm_attempts=1000, bc_max_steps=3000,
                    n_iter_tasks=32, n_eval=105, iterations=8, eval_every=4, window=512, max_state_tokens=512,
                    max_action_tokens=160, gen_action_tokens=192, vocab_cap=8000,
                    wm_backbone="answerdotai/ModernBERT-base", wm_grad_ckpt=True,
                    wm_copy_steps=1000, wm_boot_steps=1000, rep_warmup=300, wm_iter_steps=150, wm_batch=16,
                    imag_batch=32, imag_rounds=2, critic_steps=16, vm_max_tokens=1024, gen_batch=128,
                    gen_tokens=192000, lp_batch=8)
        base.update(kw)
        return Config(**base)

    @staticmethod
    def pro(**kw):
        """«Взрослый» вариант: официальный Knights & Knaves (точность по числу жителей 2–8, как в публикациях),
        актор Qwen2.5-1.5B (7B — слишком дорого при глубокой рекурсии), VM и модель мира — ModernBERT-base, окно 512.
        Датасет (ваше): все задачи обучения (6200) × 2 попытки, ≈6–9 ч (оценка), частями по ~500 попыток; судья
        сравнивает пары в фоне (≈$1). Модель мира учится на 4000 попытках из него (предл.). GSM8K:
        Config.pro(task="gsm8k")."""
        base = dict(task="kk", kk_source="hf", actor_model="Qwen/Qwen2.5-1.5B-Instruct",
                    max_steps=5, max_answer_chars=300,
                    kk_people=(2, 8), kk_eval_people=(2, 8), dataset_tasks=0, dataset_attempts=2, dataset_parts=24,
                    wm_attempts=4000, bc_max_steps=6000,
                    n_iter_tasks=64, n_eval=350, iterations=10, eval_every=5, max_state_tokens=512,
                    max_action_tokens=160, gen_action_tokens=192, window=512, vocab_cap=8000, wm_batch=16,
                    wm_backbone="answerdotai/ModernBERT-base", wm_grad_ckpt=True,
                    wm_copy_steps=2000, wm_boot_steps=2000, vm_max_tokens=1024, imag_batch=64, gen_batch=128,
                    gen_tokens=192000, lp_batch=8)
        if kw.get("task") == "gsm8k":
            base.update(max_answer_chars=40, n_eval=500, dataset_tasks=3000)
        base.update(kw)
        return Config(**base)


def fit_gpu(cfg, gb=None):
    """Настройки под память GPU (предл.): на 24 ГБ (RTX 4090) — меньше пачки генерации и обучения, контрольные точки
    градиента; на 40+ ГБ — как есть. gb — вручную (переменная WM_GPU_GB) или по первой видимой карте."""
    gb = gb or (float(os.environ.get("WM_GPU_GB", 0)) or
                (torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else 0))
    if not gb or gb >= 30 or cfg.dry_run:
        return cfg
    small = dict(gen_batch=64, gen_tokens=64000, wm_batch=8, wm_grad_ckpt=True, vm_batch=16, vm_grad_ckpt=True,
                 lp_batch=4, actor_batch=16, imag_batch=min(cfg.imag_batch, 32), critic_batch=min(cfg.critic_batch, 64))
    print(f"GPU {gb:.0f} GB: {small}")
    return replace(cfg, **small)


CFG = Config.dry() if DRY_RUN else Config.fast()

# %% [markdown]
# ## 2. Google Drive и лог

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/sleepwalker"
except ImportError:                                    # не Colab: локальная папка
    BASE_DIR = os.path.abspath("sleepwalker_runs")
if "--state-dir" in sys.argv:                               # на сервере: --state-dir ПАПКА — где лежат прогоны (состояние, логи, веса)
    BASE_DIR = os.path.abspath(os.path.expanduser(sys.argv[sys.argv.index("--state-dir") + 1]))
os.makedirs(BASE_DIR, exist_ok=True)
LOG_PATH = None
STATUS = {"path": None, "run": None, "stage": None, "started": None, "done": False, "stopped": False, "error": None,
          "lines": deque(maxlen=40)}
QUIET = "--status" in sys.argv or "--stop" in sys.argv   # служебные команды: без шапки с устройством и папкой
if not QUIET:
    print("Drive folder:", BASE_DIR)


class StopRequested(Exception):
    """Остановка по просьбе: в logs/control.json записано {"stop": true}."""


def write_status():
    """logs/status.json — состояние прогона одним файлом (предл.): этап, последние строки журнала, время, ошибка.
    Его можно скачать или прочитать, не сидя в сессии."""
    if not STATUS["path"]:
        return
    now = time.time()
    save_json({"run": STATUS["run"], "stage": STATUS["stage"], "pid": os.getpid(), "host": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "started": STATUS["started"], "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_min": round((now - STATUS["t0"]) / 60, 1) if STATUS.get("t0") else None,
               "done": STATUS["done"], "stopped": STATUS["stopped"], "error": STATUS["error"],
               "last_lines": list(STATUS["lines"])}, STATUS["path"])


def status_begin(D, name):
    STATUS.update(path=os.path.join(D.logs, "status.json"), run=name, stage="start", done=False, stopped=False,
                  error=None, started=time.strftime("%Y-%m-%d %H:%M:%S"), t0=time.time())
    remember_state_dir(BASE_DIR)
    write_status()


def stage(name):
    """Текущий этап — в status.json и в журнал."""
    STATUS["stage"] = name
    log(f"=== этап: {name}")


def check_stop(D):
    """Остановка по просьбе между частями/итерациями: logs/control.json с {"stop": true}. Файл переименовывается в
    control.done.json, чтобы следующий запуск не остановился сразу."""
    cp = os.path.join(D.logs, "control.json")
    if (load_json(cp) or {}).get("stop"):
        os.replace(cp, os.path.join(D.logs, "control.done.json"))
        raise StopRequested("остановлено по control.json")


def log(*args):
    """Печать + дозапись в log.txt (с отметкой времени) + status.json."""
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(a) for a in args)
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    STATUS["lines"].append(line[:300])
    write_status()


STATE_DIRS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
                               ".sleepwalker_state_dirs")   # какие папки состояния использовались (не в git)


def remember_state_dir(base):
    """Запомнить папку состояния: --status без --state-dir показывает прогоны из всех запомненных папок."""
    try:
        known = open(STATE_DIRS_FILE, encoding="utf-8").read().split("\n") if os.path.exists(STATE_DIRS_FILE) else []
        if base not in known:
            with open(STATE_DIRS_FILE, "a", encoding="utf-8") as f:
                f.write(base + "\n")
    except OSError:
        pass


def known_state_dirs():
    out = [BASE_DIR]
    if os.path.exists(STATE_DIRS_FILE):
        out += [d for d in open(STATE_DIRS_FILE, encoding="utf-8").read().split("\n") if d and d not in out]
    return [d for d in out if os.path.isdir(d)]


def alive(st):
    """Жив ли процесс прогона (на этой же машине): status.json без done/error, а процесса уже нет — он умер молча."""
    if st.get("host") != socket.gethostname() or not st.get("pid"):
        return None
    try:
        os.kill(int(st["pid"]), 0)
        return True
    except OSError:
        return False


def print_status(dirs=None, verbose=False):
    """python3 sleepwalker.py --status [--state-dir ПАПКА] [-v]: одна строка на прогон — состояние, этап, когда
    обновлялся; без --state-dir — все запомненные папки состояния."""
    for base in dirs or known_state_dirs():
        rows = []
        for name in sorted(os.listdir(base)):
            if name.startswith("dry_") and not verbose:   # прогоны проверки — только с -v
                continue
            st = load_json(os.path.join(base, name, "logs", "status.json"))
            if st:
                rows.append((name, st))
        if not rows:
            continue
        print(f"{base}")
        for name, st in rows:
            if st.get("error"):
                state = "ОШИБКА"
            elif st.get("stopped"):
                state = "остановлен"
            elif st.get("done"):
                state = "готово"
            else:
                state = "ИДЁТ" if alive(st) is not False else "УМЕР (процесса нет)"
            try:
                age = (time.time() - time.mktime(time.strptime(st["updated"], "%Y-%m-%d %H:%M:%S"))) / 60
                age = f"{age:.0f} мин назад" if age < 120 else f"{age / 60:.1f} ч назад"
            except Exception:
                age = st.get("updated")
            hours = f"{(st.get('elapsed_min') or 0) / 60:.1f} ч"
            print(f"  {name:28s} {state:20s} {str(st.get('stage'))[:46]:46s} обновлено {age}, работал {hours}")
            if st.get("error"):
                print(f"  {'':28s} ошибка: {st['error'][:160]}")
            for l in st.get("last_lines", [])[-(5 if verbose else 1):]:
                print(f"  {'':28s} {l[:150]}")


def request_stop(run=None, base=None):
    """--stop [ПРОГОН]: записать {"stop": true} в logs/control.json — прогон остановится на ближайшей границе части
    или итерации (состояние целое; та же команда запуска продолжит). Без имени — всем идущим прогонам папки."""
    base = base or BASE_DIR
    names = [run] if run else [n for n in sorted(os.listdir(base))
                                if (load_json(os.path.join(base, n, "logs", "status.json")) or {}).get("stage")
                                and not (load_json(os.path.join(base, n, "logs", "status.json")) or {}).get("done")]
    for n in names:
        d = os.path.join(base, n, "logs")
        if not os.path.isdir(d):
            print(f"{n}: нет такого прогона в {base}")
            continue
        save_json({"stop": True}, os.path.join(d, "control.json"))
        print(f"{n}: остановка запрошена — сработает на ближайшей границе части датасета или итерации")


def logged_errors(fn):
    """Любая ошибка (и обрыв по Ctrl-C) — с трассировкой в log.txt на Drive, затем наружу: в скачанных логах видно,
    где и почему прогон остановился."""
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except StopRequested as e:
            STATUS["stopped"] = True
            log(f"{fn.__name__}: {e}; состояние сохранено, тот же запуск продолжит")
            raise
        except BaseException as e:
            STATUS["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            if LOG_PATH:
                log(f"ОШИБКА в {fn.__name__}: {type(e).__name__}: {str(e)[:500]}\n" + traceback.format_exc()[-4000:])
            raise
    wrapped.__name__ = fn.__name__
    return wrapped


class Progress:
    """Строка в лог не чаще раза в every секунд: долгие этапы видны по ходу."""
    def __init__(self, name, total, every=180):
        self.name, self.total, self.every = name, total, every
        self.t0 = self.last = time.time()

    def tick(self, done, extra=""):
        now = time.time()
        if now - self.last < self.every:
            return
        self.last, mins = now, (now - self.t0) / 60
        # оценка остатка — только после 20 %: эпизоды кончаются пачкой к концу, и ранняя оценка сильно завышена
        eta = f", осталось ≈{mins * (self.total - done) / done:.0f} мин" if 0.2 * self.total <= done < self.total else ""
        log(f"  … {self.name}: {done}/{self.total}{extra}, прошло {mins:.0f} мин{eta}")


def tmp_name(path):
    """Своё имя временного файла у каждого процесса и потока: две записи одного файла не мешают друг другу."""
    return f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"


def save_json(obj, path):
    """Атомарная запись: обрыв посреди записи не портит прежний файл."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = tmp_name(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def save_json_gz(obj, path):
    """То же, сжато (gzip): траектории."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = tmp_name(path)
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_json_gz(path, default=None):
    if not os.path.exists(path):
        return default
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_torch(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = tmp_name(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def half(sd):
    """Вещественные тензоры — в bf16: чекпойнт модели мира вдвое меньше (при загрузке вернутся в fp32). Связанные
    тензоры (одна память под двумя именами) конвертируются один раз — torch.save хранит их один раз."""
    seen, out = {}, {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            key = (v.data_ptr(), tuple(v.shape), tuple(v.stride()))
            if key not in seen:
                seen[key] = v.to(torch.bfloat16)
            out[k] = seen[key]
        else:
            out[k] = v
    return out


# %% [markdown]
# ## 3. Задачи
# * **arith** — вложенная арифметика (генератор).
# * **kk** — «рыцари и лжецы». По умолчанию в «взрослом» варианте — официальный набор Knights & Knaves
#   (HF: K-and-K/knights-and-knaves; на нём оценивали модели исходная статья 2024 г. и Logic-RL 2025 г.), точность
#   считается по числу жителей (2–8), как в этих статьях. Если набор не загрузится — наш генератор в том же формате
#   (тогда прямого сравнения с публикациями нет; это пишется в лог). Естественное разбиение — разбор случаев.
# * **gsm8k** — школьные текстовые задачи (openai/gsm8k, скачивается в Colab библиотекой datasets).
#
# Базовая линия «прямой ответ» — та же модель с обычным пошаговым рассуждением и строкой ANSWER в конце
# (близко к постановке публикаций; точные подсказки там другие, поэтому сравнение приблизительное).
#
# Истина нужна только наверху (награда R). Для подзапросов арифметики значение считается лишь для диагностики.
# К запросам SUB в kk и gsm8k условие задачи прикладывается автоматически (предл.: иначе актору пришлось бы
# переписывать длинное условие в каждый запрос).

# %%
def parse_number(text):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text or "")
    if not m:
        return None
    try:
        v = float(m.group(0).replace(",", ""))
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def safe_eval(text):
    """Значение выражения из цифр, + - * и скобок; иначе None (диагностика и заглушки)."""
    text = (text or "").strip().rstrip(".").replace("×", "*").replace("−", "-")
    text = re.sub(r"^(compute|calculate|evaluate)\s*:?\s*", "", text, flags=re.I)
    if not text or len(text) > 200 or not re.fullmatch(r"[\d\s+\-*()]+", text):
        return None
    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            a, b = ev(n.left), ev(n.right)
            if abs(a) > 10 ** 9 or abs(b) > 10 ** 9:
                raise ValueError
            return a + b if isinstance(n.op, ast.Add) else a - b if isinstance(n.op, ast.Sub) else a * b
        raise ValueError

    try:
        return ev(node)
    except (ValueError, RecursionError):
        return None


PATH_HEAD = "PATH (read-only context from the levels above):"   # путь к шагу (ваше): контекст уровней выше
PATH_CHARS = 3000                                                 # длинный путь усекается (приоритеты — trim_path)
Q_SUB = "Q (answer only this question):"                          # и единственный вопрос, на который отвечать


class ArithTask:
    name = "arith"
    hint = ("The task Q is an arithmetic expression; the answer is a number. A larger bracket can be handed to a "
            "helper as one request, e.g. 'SUB: (12 - 5) * (3 + 4)'.")
    shots = [   # примеры для стартовой политики: реплики чата «состояние → ответ актора»
        ("Q: Compute (3 + 4) * (10 - 2)\nL: 0/{L} T: 0/{T}",
         "PLAN: compute both brackets, then multiply\nLLM: 3 + 4\nLLM: 10 - 2"),
        ("Q: Compute (3 + 4) * (10 - 2)\nL: 0/{L} T: 1/{T}\nP: compute both brackets, then multiply\n"
         "S1: LLM 3 + 4 = 7 ; LLM 10 - 2 = 8", "PLAN: multiply the results\nLLM: 7 * 8"),
        ("Q: Compute (3 + 4) * (10 - 2)\nL: 0/{L} T: 2/{T}\nP: multiply the results\n"
         "S1: LLM 3 + 4 = 7 ; LLM 10 - 2 = 8\nS2: LLM 7 * 8 = 56", "PLAN: done\nANSWER: 56"),
        (PATH_HEAD + "\n[L0] Q: Compute ((3 + 4) * (10 - 2)) - 5\n" + Q_SUB + " (3 + 4) * (10 - 2)\nL: 1/{L} T: 0/{T}",
         "PLAN: compute both brackets\nLLM: 3 + 4\nLLM: 10 - 2"),
    ]
    hint_nosub = "The task Q is an arithmetic expression; the answer is a number."
    system_exec = ("Answer only the QUESTION (an arithmetic expression); the CONTEXT is for reference. "
                   "Reply with only the resulting number.")
    system_direct = "Compute the expression step by step. End with a line 'ANSWER: <number>'."
    exec_max_new = 16
    direct_max_new = 384

    @staticmethod
    def gen_expr(rng, depth, leaf_max):
        """Случайное выражение глубины depth: (текст, значение)."""
        if depth == 0:
            v = rng.randint(2, leaf_max)
            return str(v), v
        for _ in range(50):
            lt, lv = ArithTask.gen_expr(rng, depth - 1, leaf_max)
            rt, rv = ArithTask.gen_expr(rng, rng.randint(0, depth - 1), leaf_max)
            if rng.random() < 0.5:
                lt, lv, rt, rv = rt, rv, lt, lv
            op = rng.choice(("+", "-", "*"))
            if op == "*" and abs(lv) * abs(rv) > 20000:
                op = rng.choice(("+", "-"))
            v = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
            if abs(v) <= 10 ** 5:
                wrap = lambda t: t if re.fullmatch(r"-?\d+", t) else f"({t})"
                return f"{wrap(lt)} {op} {wrap(rt)}", v
        return ArithTask.gen_expr(rng, 0, leaf_max)

    def make(self, n, seed, cfg, split="train"):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            d = cfg.depth if rng.random() < 0.7 else max(1, cfg.depth - 1)
            text, value = self.gen_expr(rng, d, cfg.leaf_max)
            out.append({"query": f"Compute {text}", "truth": value})
        return out

    def check(self, answer, truth):
        return parse_number(answer) == truth

    def norm_result(self, text):
        n = parse_number(text)
        return str(n) if n is not None else (((text or "").strip().splitlines() or ["?"])[0][:30] or "?")

    def sub_truth(self, request):
        return safe_eval(request)

    def compact(self, text):
        """Текст для окна (фаза копирования): как есть."""
        return text

    def seed_texts(self, cfg):
        rng = random.Random(123)
        return [" ".join(str(i) for i in range(-1000, 1001))] + \
            [self.gen_expr(rng, rng.randint(1, cfg.depth + 1), cfg.leaf_max)[0] for _ in range(3000)]

    def stub_action(self, text, rng, cfg):
        own = own_question(text)                         # свой вопрос, а не путь
        exprs = re.findall(r"\((\d+ [+\-*] \d+)\)", own) or re.findall(r"(\d+ [+\-*] \d+)", own)
        nums = re.findall(r"= (-?\d+)", "\n".join(x for x in text.split("\n") if x.startswith("S")))
        r = rng.random()
        if (nums and r < 0.35) or r < 0.08:
            v = safe_eval(own) if rng.random() < 0.5 else None   # иногда «решает» (разброс R для GRPO)
            return f"PLAN: done\nANSWER: {v if v is not None else (nums[-1] if nums else rng.randint(1, 99))}"
        exprs = exprs or [f"{rng.randint(2, 9)} + {rng.randint(2, 9)}" for _ in range(rng.randint(1, cfg.max_reqs))]
        lines = ["PLAN: compute parts"]
        for e in exprs[:rng.randint(1, cfg.max_reqs)]:
            lines.append(("SUB: " if rng.random() < 0.25 else "LLM: ") + e)
        return "\n".join(lines)

    def stub_exec(self, request, rng):
        v = safe_eval(question_of(request))
        if v is not None and rng.random() < 0.1:
            v += rng.choice((-1, 1))
        return str(v) if v is not None else "?"


KK_NAMES = ["Zoey", "Ethan", "Mia", "Liam", "Ava", "Noah", "Emma", "Lucas", "Olivia", "Mason", "Sophia", "Logan",
            "Isabella", "James", "Amelia", "Jack", "Harper", "Owen", "Ella", "Henry", "Chloe", "Leo", "Grace", "Ryan",
            "Lily", "Caleb", "Nora", "Isaac", "Aria", "Julian", "Riley", "Aaron", "Zoe", "David", "Hannah", "Samuel"]


class KKTask:
    name = "kk"
    hint = ("The task Q is a knights-and-knaves puzzle: knights always tell the truth, knaves always lie. A natural way "
            "to delegate is case analysis, e.g. 'SUB: Assume Zoey is a knight. Who is a knight and who is a knave?' "
            "(every SUB and LLM request automatically gets the path: the puzzle and the steps done above). The "
            "answer names the role of every "
            "inhabitant, e.g. 'ANSWER: Zoey is a knight, Ethan is a knave'. If the assumptions in Q together with "
            "those on the PATH lead to a "
            "contradiction, answer 'ANSWER: contradiction'.")
    _q = ("A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Zoey, Ethan and Mia. Zoey says: \"Ethan is a knave or Mia is a knight.\" Ethan says: \"Mia is a knight or Zoey is a knight.\" Mia says: \"Zoey is a knight if and only if Ethan is a knight.\" Who is a knight and who is a knave?")
    shots = [
        ("Q: " + _q + "\nL: 0/{L} T: 0/{T}",
         "PLAN: case analysis on Zoey\nSUB: Assume Zoey is a knight. Who is a knight and who is a knave?\n"
         "SUB: Assume Zoey is a knave. Who is a knight and who is a knave?"),
        (PATH_HEAD + "\n[L0] Q: " + _q + "\n" + Q_SUB + " Assume Zoey is a knave. Who is a knight and who is a "
         "knave?\nL: 1/{L} T: 0/{T}",
         "PLAN: Zoey lies, so Ethan is a knight and Mia is a knave; then Ethan's statement is false, impossible\n"
         "ANSWER: contradiction"),
        ("Q: " + _q + "\nL: 0/{L} T: 1/{T}\nP: case analysis on Zoey\nS1: SUB Assume Zoey is a knight. Who is a "
         "knight and who is a knave? = Zoey is a knight, Ethan is a knight, Mia is a knight ; SUB Assume Zoey is a "
         "knave. Who is a knight and who is a knave? = contradiction",
         "PLAN: only the first case is consistent\nANSWER: Zoey is a knight, Ethan is a knight, Mia is a knight"),
    ]
    hint_nosub = ("The task Q is a knights-and-knaves puzzle: knights always tell the truth, knaves always lie. "
                  "Requests get the path automatically: the puzzle and the steps done above. The answer names the role "
                  "of every inhabitant, e.g. 'ANSWER: Zoey is a knight, Ethan is a knave'. If the assumptions in Q "
                  "together with those on the PATH lead to a contradiction, answer 'ANSWER: contradiction'.")
    system_exec = ("Answer only the QUESTION about knights (always tell the truth) and knaves (always lie); the "
                   "CONTEXT holds the puzzle and the work done so far. Give only the short final answer.")
    system_direct = ("Solve the puzzle step by step. End with a line 'ANSWER: Name is a knight, Name is a knave, ...' "
                     "naming every inhabitant.")
    exec_max_new = 48
    direct_max_new = 1024

    @staticmethod
    def stmt(rng, others, depth):
        """Высказывание о других жителях: «X — рыцарь/лжец», его отрицание целиком или связка двух разных жителей."""
        if depth == 0 or len(others) < 2 or rng.random() < 0.35:
            a = ("is", rng.choice(others), rng.random() < 0.5)
            return ("not", a) if rng.random() < 0.15 else a
        x, y = rng.sample(others, 2)
        return (rng.choice(("and", "or", "if", "iff")), ("is", x, rng.random() < 0.5), ("is", y, rng.random() < 0.5))

    @staticmethod
    def ev(s, world):
        t = s[0]
        if t == "is":
            return world[s[1]] == s[2]
        if t == "not":
            return not KKTask.ev(s[1], world)
        a, b = KKTask.ev(s[1], world), KKTask.ev(s[2], world)
        return a and b if t == "and" else a or b if t == "or" else (not a) or b if t == "if" else a == b

    @staticmethod
    def text(s):
        t = s[0]
        if t == "is":
            return f"{s[1]} is a {'knight' if s[2] else 'knave'}"
        if t == "not":
            return f"it is not the case that {KKTask.text(s[1])}"
        a, b = KKTask.text(s[1]), KKTask.text(s[2])
        return {"and": f"{a} and {b}", "or": f"{a} or {b}", "if": f"if {a} then {b}",
                "iff": f"{a} if and only if {b}"}[t]

    def puzzle(self, rng, n, depth):
        """Головоломка с единственным решением: (текст, {имя: рыцарь?})."""
        for _ in range(500):
            names = rng.sample(KK_NAMES, n)
            # высказывания о других; при двух жителях (и иногда при большем числе) — и о себе, как в наборе K&K
            stmts = [self.stmt(rng, names if (n <= 2 or rng.random() < 0.2) else [x for x in names if x != me], depth)
                     for me in names]
            sols = []
            for bits in itertools.product((True, False), repeat=n):
                world = dict(zip(names, bits))
                if all(self.ev(st, world) == world[me] for me, st in zip(names, stmts)):
                    sols.append(world)
                    if len(sols) > 1:
                        break
            if len(sols) == 1:
                says = " ".join(f'{me} says: "{self.text(st)[0].upper() + self.text(st)[1:]}."'
                                for me, st in zip(names, stmts))
                q = (f"A very special island is inhabited only by knights and knaves. Knights always tell the truth, "
                     f"and knaves always lie. You meet {n} inhabitants: {', '.join(names[:-1])} and {names[-1]}. "
                     f"{says} Who is a knight and who is a knave?")
                return q, sols[0]
        raise RuntimeError("не удалось построить головоломку с единственным решением")

    _hf = None

    def load_hf(self):
        """Официальный набор K&K: {train, test} → [{query, truth, ppl}]. None — если не загрузился."""
        if KKTask._hf is None:
            err = None
            for k in range(4):                           # сеть может моргнуть: несколько попыток с паузой
                try:
                    from datasets import load_dataset
                    out = {}
                    for split in ("train", "test"):
                        rows = []
                        dd = load_dataset("K-and-K/knights-and-knaves", split)
                        for part, ds in dd.items():
                            m = re.search(r"(\d+)\s*ppl", part)
                            for r in ds:
                                names = list(r["names"])
                                truth = {nm: bool(v) for nm, v in zip(names, r["solution"])}
                                rows.append({"query": r["quiz"], "truth": truth,
                                             "ppl": int(m.group(1)) if m else len(names)})
                        out[split] = rows
                    KKTask._hf = out
                    log("K&K: официальный набор загружен:", {k: len(v) for k, v in out.items()},
                        "| жителей:", sorted({r["ppl"] for r in out["test"]}))
                    break
                except Exception as e:                   # нет сети, другое имя или формат набора
                    err = e
                    time.sleep(5 * (k + 1))
            if KKTask._hf is None:                       # не кэшируем отказ: следующий вызов попробует снова
                raise RuntimeError(f"официальный набор K&K не загрузился ({repr(err)[:200]}). Прогон не стартует: "
                                   "датасет и оценка должны быть на официальных задачах. Проверьте доступ к HF или "
                                   "задайте kk_source=\"gen\" (наш генератор; сравнения с публикациями не будет)")
        return KKTask._hf or None

    def train_count(self, cfg):
        """Сколько задач обучения в диапазоне жителей (официальный набор); у генератора — столько же, сколько в наборе."""
        data = self.load_hf() if cfg.kk_source == "hf" else None
        lo, hi = cfg.kk_people
        return len([t for t in data["train"] if lo <= t["ppl"] <= hi]) if data else 6200

    def make(self, n, seed, cfg, split="train"):
        rng = random.Random(seed)
        data = self.load_hf() if cfg.kk_source == "hf" else None
        if data:
            lo, hi = cfg.kk_people if split == "train" else cfg.kk_eval_people
            pool = [t for t in data[split] if lo <= t["ppl"] <= hi]
            if split == "train":
                return [rng.choice(pool) for _ in range(n)] if n > len(pool) else rng.sample(pool, n)
            by = {}
            for t in pool:                             # отложенный набор: поровну по числу жителей, порядок набора
                by.setdefault(t["ppl"], []).append(t)
            k = max(1, n // max(1, len(by)))
            return [t for p in sorted(by) for t in by[p][:k]]
        out = []
        lo, hi = cfg.kk_people if split == "train" else cfg.kk_eval_people
        for _ in range(n):
            ppl = rng.randint(lo, hi)
            q, sol = self.puzzle(rng, ppl, cfg.kk_depth)
            out.append({"query": q, "truth": sol, "ppl": ppl})
        return out

    @staticmethod
    def roles(answer):
        found = {}
        for name, role in re.findall(r"([A-Z][a-z]+)\s+is\s+an?\s+(knight|knave)", answer or ""):
            found.setdefault(name, role == "knight")
        return found

    def check(self, answer, truth):
        found = self.roles(answer)
        return bool(truth) and all(found.get(n) == v for n, v in truth.items())

    def norm_result(self, text):
        text = (text or "").strip()
        return text.splitlines()[0][:200] if text else "?"

    def sub_truth(self, request):
        return None

    KK_INTRO = re.compile(r"A very special island is inhabited only by knights and knaves\. Knights always tell the "
                          r"truth, and knaves always lie\. ")
    KK_ASK = re.compile(r"\s*(So )?[Ww]ho is a knight and who is a knave\?")

    def compact(self, text):
        """Текст для окна (фаза копирования, предл.): без общего вступления и вопроса — в окне мало места."""
        return self.KK_ASK.sub("", self.KK_INTRO.sub("", text))

    def seed_texts(self, cfg):
        rng = random.Random(123)
        out = ["Assume contradiction knight knave knights knaves Who is a and only if then it is not the case that"]
        data = self.load_hf() if cfg.kk_source == "hf" else None
        if data:
            for t in data["train"][:2000]:
                out += [t["query"], ", ".join(f"{k} is a {'knight' if v else 'knave'}" for k, v in t["truth"].items())]
        for _ in range(400):
            try:
                q, sol = self.puzzle(rng, rng.randint(2, 8), cfg.kk_depth)
            except RuntimeError:
                continue
            out += [q, ", ".join(f"{k} is a {'knight' if v else 'knave'}" for k, v in sol.items())]
        return out

    def stub_action(self, text, rng, cfg):
        m = re.search(r"inhabitants: ([A-Za-z, ]+?)\.", text)
        names = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)) if x] if m else KK_NAMES[:3]
        r = rng.random()
        if r < 0.35 or "=" in text and r < 0.6:
            return "PLAN: answer\nANSWER: " + ", ".join(f"{x} is a {rng.choice(('knight', 'knave'))}" for x in names)
        x = rng.choice(names)
        if rng.random() < 0.5:
            return (f"PLAN: case analysis on {x}\nSUB: Assume {x} is a knight. Who is a knight and who is a knave?\n"
                    f"SUB: Assume {x} is a knave. Who is a knight and who is a knave?")
        return f"PLAN: check {x}\nLLM: Is {x} a knight or a knave?"

    def stub_exec(self, request, rng):
        if rng.random() < 0.3:
            return "contradiction"
        names = re.findall(r"\b([A-Z][a-z]+)\b", question_of(request))
        names = [x for x in names if x not in ("Assume", "Who", "Is", "Check", "A", "Knights", "You")] or KK_NAMES[:2]
        return ", ".join(f"{x} is a {rng.choice(('knight', 'knave'))}" for x in dict.fromkeys(names))


class GSM8KTask(ArithTask):
    name = "gsm8k"
    hint = ("The task Q is a grade-school math word problem; the answer is a number. Delegate sub-questions or "
            "calculations (every SUB and LLM request automatically gets the path: the problem and the steps above).")
    _q = "Tom has 3 boxes with 12 apples in each box. He gives 10 apples to a friend. How many apples does Tom have left?"
    shots = [
        ("Q: " + _q + "\nL: 0/{L} T: 0/{T}", "PLAN: count all apples, then subtract the gift\nLLM: 3 * 12"),
        ("Q: " + _q + "\nL: 0/{L} T: 1/{T}\nP: count all apples, then subtract the gift\nS1: LLM 3 * 12 = 36",
         "PLAN: subtract the 10 apples\nLLM: 36 - 10"),
        ("Q: " + _q + "\nL: 0/{L} T: 2/{T}\nP: subtract the 10 apples\nS1: LLM 3 * 12 = 36\nS2: LLM 36 - 10 = 26",
         "PLAN: done\nANSWER: 26"),
        (PATH_HEAD + "\n[L0] Q: " + _q + "\n" + Q_SUB + " How many apples are in the 3 boxes?\nL: 1/{L} T: 0/{T}",
         "PLAN: multiply\nLLM: 3 * 12"),
    ]
    hint_nosub = ("The task Q is a grade-school math word problem; the answer is a number. Delegate calculations "
                  "(every request automatically gets the path: the problem and the steps above).")
    system_exec = ("Answer only the QUESTION; the CONTEXT holds the problem and the work done so far. "
                   "Reply with only the final number.")
    system_direct = "Solve the problem step by step. End with a line 'ANSWER: <number>'."
    exec_max_new = 24
    direct_max_new = 512
    _data = None

    def load(self):
        if GSM8KTask._data is None:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main")
            conv = lambda split: [{"query": r["question"], "truth": parse_number(r["answer"].split("####")[-1])}
                                  for r in ds[split]]
            GSM8KTask._data = {"train": conv("train"), "test": conv("test")}
        return GSM8KTask._data

    def make(self, n, seed, cfg, split="train"):
        data = self.load()[split]
        return random.Random(seed).sample(data, min(n, len(data)))

    def check(self, answer, truth):
        v = parse_number(answer)
        return v is not None and truth is not None and abs(float(v) - float(truth)) < 1e-6

    def seed_texts(self, cfg):
        return [r["query"] for r in self.load()["train"][:2000]] + [" ".join(str(i) for i in range(0, 1001))]


TASKS = {"arith": ArithTask, "kk": KKTask, "gsm8k": GSM8KTask}
TASK = None


def task_source(cfg):
    """Откуда задачи: hf — официальный набор (сравнение с публикациями), gen — наш генератор."""
    if cfg.task == "kk":
        return "hf" if cfg.kk_source == "hf" and TASK.load_hf() else "gen"
    return "hf" if cfg.task == "gsm8k" else "gen"


def set_task(cfg):
    global TASK
    TASK = TASKS[cfg.task]()
    return TASK


def make_tasks(n, seed, cfg, split="train"):
    return TASK.make(n, seed, cfg, split)


def dataset_task_count(cfg):
    """Задач в датасете: dataset_tasks, а 0 — все задачи обучения (у K&K — весь train в диапазоне жителей)."""
    if cfg.dataset_tasks > 0:
        return cfg.dataset_tasks
    f = getattr(TASK, "train_count", None)
    return f(cfg) if f else 1000


def check_answer(answer, truth):
    return answer is not None and truth is not None and TASK.check(answer, truth)


# %% [markdown]
# ## 4. Тексты: состояние, действие, подсказки
# Состояние s = запрос + план + все шаги с результатами (ваше). Действие актора — строки PLAN / LLM / SUB / ANSWER.

# %%
def step_line(i, st):
    """Шаг в состоянии: запросы — строкой S<i>, их результаты — отдельной строкой R<i>, в том же порядке (предл.).
    Раньше было «S1: SUB вопрос = результат»: модель копировала этот образец и сама дописывала «= результат» к своим
    запросам (62 % запросов в датасете pro_v7); с результатами в отдельной строке образца для копирования нет
    (на моделях-заместителях доля таких запросов упала с 20–36 % до 1 %)."""
    if st.get("answer") is not None:
        return f"S{i}: ANSWER {st['answer']}"
    if not st["reqs"]:
        return f"S{i}: -"
    return (f"S{i}: " + " ; ".join(f"{r['dest']} {r['text']}" for r in st["reqs"]) + "\n" +
            f"R{i}: " + " ; ".join(str(r.get("result", "?")) for r in st["reqs"]))


def clean_request(text):
    """Текст запроса без дописанного моделью «= результат» и без второго запроса через « ; » (см. parse_action)."""
    return re.split(r"\s=\s|\s;\s", text, 1)[0].strip() or text


def strip_echo(question, result):
    """Исполнитель часто повторяет вопрос перед ответом («вопрос = ответ», «вопрос ответ»; в датасете pro_v7 — 22 %
    ответов LLM): оставить только ответ. Если кроме повтора ничего нет — «?» (запрос не дал ответа)."""
    q, r = (question or "").strip(), (result or "").strip()
    if q and r.lower().startswith(q.lower()):
        r = r[len(q):].strip().lstrip("=:-—").strip()
        return r or "?"
    return result


def clean_action(text):
    """Действие без дописанных моделью «= результат» в строках запросов (для π0: не учить её этому мусору)."""
    out = []
    for line in (text or "").split("\n"):
        m = ACT_RE.match(line)
        if m and m.group(1).upper() in ("SUB", "LLM"):
            out.append(f"{m.group(1)}: {clean_request(m.group(2))}")
        else:
            out.append(line)
    return "\n".join(out)


def shot_state(u):
    """Состояние примера из прежней записи «S1: LLM q = r ; SUB q2 = r2» — в текущую (S<i> и R<i>)."""
    out = []
    for line in u.split("\n"):
        m = re.match(r"^(S\d+): (.*)$", line)
        if m and " = " in line and not m.group(2).startswith("ANSWER"):
            qs, rs = zip(*[p.partition(" = ")[::2] for p in m.group(2).split(" ; ")])
            out += [f"{m.group(1)}: " + " ; ".join(qs), f"R{m.group(1)[1:]}: " + " ; ".join(rs)]
        else:
            out.append(line)
    return "\n".join(out)


def trim_path(path, max_chars):
    """Путь с приоритетами (если длинный): исходная задача, вопросы уровней ([Lk] Q — предпосылки), затем шаги
    уровней — сначала свежие. Порядок строк сохраняется; пропуски помечены «[...]»."""
    if sum(len(x) + 1 for x in path) <= max_chars:
        return list(path)
    order = [0] + [i for i, x in enumerate(path) if i and "] Q: " in x] + \
        [i for i in range(len(path) - 1, 0, -1) if "] Q: " not in path[i]]
    keep, room = set(), max_chars
    for i in order:
        if len(path[i]) + 1 <= room:
            keep.add(i)
            room -= len(path[i]) + 1
    out = []
    for i, x in enumerate(path):
        if i in keep:
            out.append(x)
        elif not out or out[-1] != "[...]":
            out.append("[...]")
    return out


def context_block(f, si):
    """Путь к шагу si кадра f (ваше): путь кадра + его вопрос + шаги до si с результатами (по уровням)."""
    return list(f.get("path", ())) + [f"[L{f['level']}] Q: {f['request']}"] + \
        [f"[L{f['level']}] {x}" for i, st in enumerate(f["steps"][:si], 1) for x in step_line(i, st).split("\n")]


def exec_content(f, si, text):
    """Запрос исполнителю: путь к шагу и сам вопрос, явно помеченный как единственный, на который отвечать."""
    return ("CONTEXT (read-only):\n" + "\n".join(trim_path(context_block(f, si), PATH_CHARS)) +
            f"\nQUESTION (answer only this): {text}")


def own_question(state):
    """Свой вопрос кадра из текста состояния (строка «Q (answer only this question)» или «Q:»)."""
    for line in state.split("\n"):
        if line.startswith(Q_SUB):
            return line[len(Q_SUB):].strip()
        if line.startswith("Q: "):
            return line[3:].strip()
    return state


def question_of(text):
    """Сам вопрос из запроса исполнителю (без контекста)."""
    return text.rsplit("QUESTION (answer only this): ", 1)[-1]


def render_state(request, level, plan, steps, cfg, path=(), subs=None):
    lines = [PATH_HEAD, *trim_path(path, PATH_CHARS), f"{Q_SUB} {request}"] if path else [f"Q: {request}"]
    lines.append(f"L: {level}/{cfg.max_level} T: {len(steps)}/{cfg.max_steps}" +
                 (f" SUB: {subs}" if subs is not None else ""))
    if plan:
        lines.append(f"P: {plan}")
    lines += [step_line(i, st) for i, st in enumerate(steps, 1)]
    return "\n".join(lines)


def vm_text(f, keep=None):
    """Вход VM: путь к кадру (контекст) + оцениваемый запрос, явно помеченный, + траектория уровня (keep — какие
    шаги оставить, для коалиций; путь не трогается)."""
    lines = [f"REQUEST (evaluate the trajectory for this request only): {f['request']}"]   # первым: не отрежется
    if f.get("path"):
        lines += ["PATH (context):", *trim_path(f["path"], PATH_CHARS // 2)]
    lines += [step_line(i, st) for i, st in enumerate(f["steps"], 1) if keep is None or (i - 1) in keep]
    return "\n".join(lines)


ACT_RE = re.compile(r"^\s*(PLAN|LLM|SUB|ANSWER)\s*:\s*(.*?)\s*$", re.I)
STATE_LINE_RE = re.compile(r"^\s*S\d+\s*:\s*(.*?)\s*$", re.I)        # модель пишет в стиле состояния: «S2: SUB q ; LLM q2»
STATE_REQ_RE = re.compile(r"^(SUB|LLM|ANSWER)\b\s*:?\s*(.*?)\s*$", re.I)


def parse_action(text, level, cfg):
    """(план, запросы, ответ, игроки). SUB на последнем уровне исполняется как LLM. Игроки — строки действия, как
    они написаны: PLAN, запросы, ANSWER (для Shapley актора: ваше «игроки — действия без результатов»; строка PLAN
    тоже игрок — предл.)."""
    plan, reqs, answer, plan_line, answer_line = None, [], None, None, None
    for line in (text or "").splitlines():
        m = ACT_RE.match(line)
        if m:
            items = [(m.group(1).upper(), m.group(2)[:200])]
        else:                                           # строка в стиле состояния «S2: SUB q ; LLM q2» — тоже запросы
            ms = STATE_LINE_RE.match(line)              # (модель так продолжает состояние вместо действия)
            parts = [STATE_REQ_RE.match(x.strip()) for x in ms.group(1).split(" ; ")] if ms else []
            items = [(x.group(1).upper(), x.group(2)[:200]) for x in parts if x]
            if not items:
                continue
        for key, val in items:
            if key == "PLAN":
                plan, plan_line = val, line.strip()
            elif key == "ANSWER":
                answer, answer_line = val[:cfg.max_answer_chars], line.strip()
                break
            elif val and len(reqs) < cfg.max_reqs:
                dest = "SUB" if key == "SUB" and level < cfg.max_level else "LLM"
                # модель часто дописывает к запросу «= результат» — подражая формату состояния; результат даёт только
                # исполнение, поэтому хвост отрезается (в датасете pro_v7 так было у 62 % запросов SUB)
                val = re.split(r"\s=\s|\s;\s", val, 1)[0].strip()
                if not val or any(r["text"] == val for r in reqs):   # повтор того же запроса — не нужен
                    continue
                reqs.append({"dest": dest, "text": val, "line": line.strip()})
        if answer is not None:
            break
    if answer is not None:
        reqs = []
    players = ([plan_line] if plan_line else []) + [r["line"] for r in reqs] + ([answer_line] if answer_line else [])
    return plan, reqs, answer, players


ACTOR_INTRO = ("You solve the task Q of the state (at sub-levels: the line 'Q (answer only this question)') by "
               "delegating. Each turn write 'PLAN: <short plan>' and then either up to {k} request lines ")
ACTOR_REQS_SUB = ("'LLM: <request>' (an assistant answers it in one go) or 'SUB: <sub-task>' (a helper solves it in "
                  "several turns, the same way you do; SUB in the state shows how many sub-tasks you may still create)")
ACTOR_REQS_LLM = "'LLM: <request>' (an assistant answers it in one go)"
ACTOR_REST = (", or a line 'ANSWER: <final answer>'. Requests of one turn run in parallel and do not see each other's "
              "results. Every request automatically gets the path: the task and the steps done so far on every level; "
              "anything else it needs must be written into the request. The requests of turn n appear in the state as S<n> "
              "and their results, in the same order, as R<n>. "
              "T shows used/available turns: answer before they run out. If the state starts with PATH, those lines "
              "are read-only context from the levels above: answer only the question marked "
              "'Q (answer only this question)'.")
SYSTEM_ACTOR = ACTOR_INTRO + ACTOR_REQS_SUB + ACTOR_REST


def sub_allow(cfg, level):
    """Сколько подзадач может создать задача уровня level (ваше): 16, 8, 4, дальше по 2; на последнем уровне — 0."""
    return max(cfg.sub_min, cfg.sub_top >> level) if level < cfg.max_level else 0


def actor_system(cfg, allow_sub=True):
    """Подсказка актора. Счётчик подзадач кончился — SUB в ней даже не предлагается (ваше)."""
    reqs = ACTOR_REQS_SUB if allow_sub else ACTOR_REQS_LLM
    return (ACTOR_INTRO.format(k=cfg.max_reqs) + reqs + ACTOR_REST + " " +
            (TASK.hint if allow_sub else TASK.hint_nosub))


def task_shots(cfg, allow_sub=True):
    """Примеры с настоящими L, T и остатком SUB; без SUB — только примеры, где SUB нет."""
    out = []
    for u, a in TASK.shots:
        if not allow_sub and "SUB:" in a:
            continue
        lvl = int(re.search(r"L: (\d+)/\{L\}", u).group(1))
        used = sum(x.count("SUB ") for x in u.split("\n") if re.match(r"S\d+:", x))
        left = max(0, sub_allow(cfg, lvl) - used) if allow_sub else 0
        u = shot_state(u.replace("/{T}", "/{T} SUB: {S}", 1))
        out.append((u.replace("{L}", str(cfg.max_level)).replace("{T}", str(cfg.max_steps)).replace("{S}", str(left)),
                    a))
    return out


def actor_prompt(actor, cfg, content, few_shot=False, allow_sub=True):
    """Подсказка актора: система + (примеры репликами чата — только для стартовой политики) + состояние или окно."""
    return actor.prompt_ids(actor_system(cfg, allow_sub), "STATE:\n" + content,
                            task_shots(cfg, allow_sub) if few_shot else None)


# модель не должна дописывать следующее состояние: в стартовых данных pro_v2 так продолжались ~4,5 % действий
# (строки «S1: … = …» с выдуманными результатами, «L: 1/1 T: 1/5»)
STOP_STRINGS = ["STATE:", "REPLY:", "\nQ:", "\nL: ", "\nP: ", "\nT: ", "\nPATH", "\n[L"] + \
    [f"\nR{i}:" for i in range(1, 10)]                # «S2: SUB q» модель пишет вместо действия — это разбирается как
#                                                       запросы; выдуманные результаты «R2: …» обрезаются


# %% [markdown]
# ## 5. Токены модели мира
# Окно — токены словаря актора, поэтому актор читает его как текст. Модель мира работает в компактном словаре:
# только токены, которые встречаются в данных (предл.; иначе слой эмбеддингов на 150 тыс. токенов).

# %%
class CharTok:
    """Токенизатор-заглушка для DRY_RUN: по символу."""
    def __init__(self):
        self.itos = ["<pad>", "<unk>", "<eos>"] + [chr(i) for i in range(32, 127)] + ["\n"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_token_id, self.eos_token_id = 0, 2

    def encode(self, text):
        return [self.stoi.get(c, 1) for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i > 2)


class HFTok:
    """Обёртка токенизатора HF: encode/decode без служебных токенов."""
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


class Vocab:
    PAD, MASK, UNK, SEP, EMPTY = 0, 1, 2, 3, 4      # EMPTY — пустая ячейка окна (её актор не видит)
    N_SPECIAL = 5

    def __init__(self, base_ids):
        self.itob = [int(b) for b in base_ids]
        self.btoi = {b: i + self.N_SPECIAL for i, b in enumerate(self.itob)}

    @property
    def size(self):
        return len(self.itob) + self.N_SPECIAL

    def to_compact(self, ids):
        return [self.btoi.get(b, self.UNK) for b in ids]

    def to_base(self, cids):
        return [self.itob[c - self.N_SPECIAL] for c in cids if c >= self.N_SPECIAL]

    @staticmethod
    def build(texts, tok, cap):
        cnt = Counter()
        for t in texts:
            cnt.update(tok.encode(t))
        return Vocab([b for b, _ in cnt.most_common(cap)])


class Codec:
    """Текст ↔ компактные id (с кэшем); окно → текст для актора."""
    def __init__(self, tok, vocab):
        self.tok, self.vocab = tok, vocab
        self._cache = {}

    def ids(self, text, max_len, keep_head=48):
        key = (text, max_len)
        if key not in self._cache:
            c = self.vocab.to_compact(self.tok.encode(text)) or [Vocab.SEP]
            if len(c) > max_len:                       # запрос — в начале, свежие шаги — в конце
                c = c[:keep_head] + c[-(max_len - keep_head):]
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = c
        return self._cache[key]

    def window_text(self, wids):
        return self.tok.decode(self.vocab.to_base(wids))

    def layout(self, text, budget):
        """Сжатое состояние в budget токенов (предл.): без общего вступления задачи (TASK.compact); запрос — не больше
        max(budget/2, budget − 64), строка «L: … T: …», план, затем свежие шаги с конца. Так условие задачи не режется
        ни во входе модели мира, ни в окне. Служебные id (<unk> и т. п.) выбрасываются."""
        enc = lambda t: [x for x in self.vocab.to_compact(self.tok.encode(t)) if x >= Vocab.N_SPECIAL]
        nl = enc("\n")
        lines = TASK.compact(text).split("\n")
        k = next((i for i, x in enumerate(lines) if x.startswith("L: ")), len(lines) - 1)
        cap = max(budget // 2, budget - 64)
        if lines[0] == PATH_HEAD and k >= 2:
            # подзадача: на путь и свой вопрос — не больше 5/8 бюджета, остальное — строка ходов, план и свои шаги.
            # Приоритет: свой вопрос, исходная задача, вопросы уровней ([Lk] Q), затем шаги уровней — свежие первыми
            cap = budget * 5 // 8
            own, hdr = enc(lines[k - 1])[:cap // 4], enc(PATH_HEAD)
            root = enc(lines[1])[:max(0, cap * 5 // 8 - len(own) - len(hdr))]
            room = cap - len(own) - len(hdr) - len(root) - 3 * len(nl)
            mid = lines[2:k - 1]
            chosen = {}
            order = [i for i, x in enumerate(mid) if "] Q: " in x] + \
                [i for i in range(len(mid) - 1, -1, -1) if "] Q: " not in mid[i]]
            for i in order:                             # не влезла строка — пробуем следующие (continue, не break)
                e = enc(mid[i]) + nl
                if len(e) <= room:
                    chosen[i], room = e, room - len(e)
            query = hdr + nl + root + nl + [t for i in sorted(chosen) for t in chosen[i]] + own
        else:
            query = enc("\n".join(lines[:k]))
        head = query[:cap] + nl + enc(lines[k]) + nl
        rest = lines[k + 1:]
        if rest and rest[0].startswith("P: "):
            head += enc(rest[0])[:budget // 8] + nl
            rest = rest[1:]
        head = head[:budget]
        tail = []
        for x in reversed(rest):
            room = budget - len(head) - len(tail)
            if room <= 0:
                break
            tail = (enc(x) + nl)[-room:] + tail
        return head + tail

    def state_ids(self, text, cfg):
        """Вход модели мира для состояния (с кэшем)."""
        key = (text, "state", cfg.max_state_tokens)
        if key not in self._cache:
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = self.layout(text, cfg.max_state_tokens) or [Vocab.SEP]
        return self._cache[key]

    def copy_target(self, text, cfg):
        """Цель копирования (предл.): раскладка состояния в W токенов, остаток — EMPTY."""
        c = self.layout(text, cfg.window)
        return c + [Vocab.EMPTY] * (cfg.window - len(c))


def wm_tok(cfg):
    """Токенизатор модели мира: её собственный (окно — её токены; актор читает окно как текст)."""
    if cfg.dry_run:
        return CharTok()
    from transformers import AutoTokenizer
    return HFTok(AutoTokenizer.from_pretrained(cfg.wm_backbone))


def vocab_seed_texts(cfg):
    """Тексты для словаря: служебные строки и много арифметики, чтобы покрыть все числа и операции."""
    rng = random.Random(123)
    texts = [SYSTEM_ACTOR, TASK.hint, "PLAN: LLM: SUB: ANSWER: NO ANSWER Q: L: T: P: = ? ; -", PATH_HEAD, Q_SUB,
             "REQUEST (evaluate the trajectory for this request only): PATH (context): CONTEXT (read-only): "
             "QUESTION (answer only this): " + " ".join(f"[L{i}]" for i in range(17))]
    texts += [u + "\n" + a for u, a in task_shots(cfg)]
    texts += TASK.seed_texts(cfg)
    for i in range(1, cfg.max_steps + 3):
        texts.append(f"S{i}: L: {i}/{cfg.max_level} T: {i}/{cfg.max_steps}")
    return texts


def pad_batch(seqs, pad=Vocab.PAD):
    """Список последовательностей → (id [B,L], маска паддинга [B,L], True = паддинг)."""
    seqs = [s if len(s) else [Vocab.SEP] for s in seqs]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    m = torch.ones((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        m[i, :len(s)] = False
    return ids.to(DEVICE), m.to(DEVICE)


# %% [markdown]
# ## 6. Модель мира: энкодер, декодер, RSSM (денойзеры), RM и критик
# * Энкодер: пустое окно + s → окно l (один проход). Декодер: l + зашумлённый (не пустой) s → s, на нескольких уровнях
#   шума. RSSM: [l_t, a_t] + зашумлённое окно l_{t+1} → l_{t+1}; при предсказании — несколько шагов очистки.
# * Градиент через дискретное окно — STE: вперёд выбранный токен, назад смесь top-k (ваше).
# * Потери как в DreamerV3: декодер + 0.5·KL(sg(энкодер) ‖ RSSM) + 0.1·KL(энкодер ‖ sg(RSSM)) — потеря RSSM идёт
#   и в энкодер (ваше, W2).
# * Против схлопывания окна (предл.; в full_v1 окно ничего не несло): сначала энкодер учится копировать в окно сжатый
#   текст состояния; затем вес KL(энкодер ‖ RSSM) растёт от 0, а копирование остаётся с малым весом.
# * Все части окна стоят в начале входа: позиции окна не зависят от длины остального текста в пачке.

# %%
SEG_STATE, SEG_WIN_OUT, SEG_WIN_IN, SEG_ACTION = 0, 1, 2, 3


class Denoiser(nn.Module):
    """Двунаправленный трансформер: части (токены или эмбеддинги, сегмент, паддинг) → логиты по словарю."""
    def __init__(self, V, d, layers, heads, max_len):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(max_len, d)
        self.seg = nn.Embedding(4, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    @property
    def emb_weight(self):
        return self.tok.weight

    def forward(self, parts, out=slice(None)):
        """out — какие позиции вернуть (логиты считаются только для них)."""
        xs, pads, segs = [], [], []
        for x, seg, pad in parts:
            xs.append(self.tok(x) if x.dtype == torch.long else x)
            pads.append(pad)
            segs.append(torch.full(pad.shape, seg, dtype=torch.long, device=pad.device))
        x, pad, seg = torch.cat(xs, 1), torch.cat(pads, 1), torch.cat(segs, 1)
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.body(x + self.pos(pos)[None] + self.seg(seg), src_key_padding_mask=pad)[:, out]
        # общие веса входа и выхода; деление на √d (как в T5): без него логиты в начале ~√d ≈ 23 при d = 512,
        # распределение окна — почти «один токен», и окно не учится
        return (self.norm(h) @ self.tok.weight.t()) * self.tok.weight.shape[1] ** -0.5


class MLMDenoiser(nn.Module):
    """Денойзер на предобученной BERT-подобной модели (ваше: денойзеры — BERT; модель — cfg.wm_backbone, предл.):
    знает язык с самого начала, на формат дообучается. Работает в компактном словаре модели мира: id компактного
    словаря → id её токенизатора (вход), логиты — только по токенам компактного словаря и только на нужных позициях.
    Части входа различаются своими обучаемыми эмбеддингами сегментов (с нуля; добавляются ко входу)."""
    checked = set()                                       # имена моделей, у которых голова уже сверена с HF

    def __init__(self, name, vocab, grad_ckpt=False, max_part=1024):
        super().__init__()
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        mc = AutoConfig.from_pretrained(name)
        if hasattr(mc, "reference_compile"):              # ModernBERT: без torch.compile внутри
            mc.reference_compile = False
        self.mlm, err = None, None
        for kw in ({"dtype": torch.float32, "attn_implementation": "sdpa"}, {"torch_dtype": torch.float32},
                   {}):                                   # веса — в fp32 (обучение под autocast bf16)
            try:
                self.mlm = AutoModelForMaskedLM.from_pretrained(name, config=mc, **kw)
                break
            except (TypeError, ValueError) as e:
                err = e
        if self.mlm is None:
            raise err
        self.mlm = self.mlm.float()
        if grad_ckpt:                                     # длинные входы: память на активации
            self.mlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        sp = [tok.pad_token_id, tok.mask_token_id, tok.unk_token_id, tok.sep_token_id, tok.pad_token_id]
        assert len(sp) == Vocab.N_SPECIAL and None not in sp, f"у {name} нет нужных служебных токенов"
        self.register_buffer("ids", torch.tensor(sp + list(vocab.itob), dtype=torch.long), persistent=False)
        self.bos = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id   # <s>, как при предобучении
        d = self.mlm.get_input_embeddings().weight.shape[1]
        self.seg = nn.Embedding(4, d)
        nn.init.zeros_(self.seg.weight)                   # в начале — ровно предобученная модель
        self.limit = self.mlm.config.max_position_embeddings - 2
        lm = getattr(self.mlm, "lm_head", None)
        self.roberta_head = lm is not None and all(hasattr(lm, a) for a in ("dense", "layer_norm", "decoder"))
        self.modernbert_head = hasattr(self.mlm, "head") and hasattr(self.mlm, "decoder")
        if "modernbert" in name.lower():
            assert self.modernbert_head, "у ModernBertForMaskedLM нет head/decoder — другая версия transformers"
        if "roberta" in name.lower():
            assert self.roberta_head, "у RobertaForMaskedLM нет lm_head.dense/layer_norm/decoder"
        # позиции внутри каждой части входа (окно, состояние, действие), с нуля: ячейка окна i и токен состояния i
        # получают одну метку — копировать проще (у ModernBERT только относительные позиции и много локальных слоёв)
        self.ppos = nn.Embedding(max_part, d)
        nn.init.zeros_(self.ppos.weight)
        if name not in MLMDenoiser.checked:              # своя голова должна совпадать с головой самой модели
            self.head_check(tok)
            MLMDenoiser.checked.add(name)

    @property
    def emb_weight(self):
        return self.mlm.get_input_embeddings().weight[self.ids]

    def head_logits(self, h, cols):
        """Голова MLM на скрытых состояниях h — только по столбцам словаря cols."""
        if self.roberta_head:                             # RoBERTa: dense → gelu → layer_norm → decoder
            lm = self.mlm.lm_head
            z = lm.layer_norm(F.gelu(lm.dense(h)))
            bias = lm.decoder.bias if lm.decoder.bias is not None else lm.bias
            return z @ lm.decoder.weight[cols].t() + bias[cols]
        if self.modernbert_head:                          # ModernBERT: head (dense → act → norm) → decoder
            dec = self.mlm.decoder
            out = self.mlm.head(h) @ dec.weight[cols].t()
            return out + dec.bias[cols] if dec.bias is not None else out
        return self.mlm.get_output_embeddings()(h)[..., cols]

    @torch.no_grad()
    def head_check(self, tok):
        """Наш путь (inputs_embeds → base_model → голова) против forward самой модели на одной фразе."""
        was = self.mlm.training
        self.mlm.eval()
        ids = torch.tensor([tok("Zoey says that Oliver is a knave.\nL: 1/16 T: 0/5").input_ids])
        ref = self.mlm(input_ids=ids).logits.float()
        h = self.mlm.base_model(inputs_embeds=self.mlm.get_input_embeddings()(ids),
                                attention_mask=torch.ones_like(ids)).last_hidden_state
        mine = self.head_logits(h, torch.arange(ref.shape[-1])).float()
        diff = float((ref - mine).abs().max())
        self.mlm.train(was)
        log(f"модель мира: своя голова против forward модели — расхождение логитов {diff:.1e}")
        assert diff < 1e-2, f"путь через inputs_embeds не совпадает с моделью ({diff:.3f}) — другая версия transformers?"

    def forward(self, parts, out=slice(None)):
        emb = self.mlm.get_input_embeddings()
        xs, pads, segs = [], [], []
        for x, seg, pad in parts:
            e = emb(self.ids[x]) if x.dtype == torch.long else x
            xs.append(e + self.ppos(torch.arange(e.shape[1], device=e.device))[None].to(e.dtype))
            pads.append(pad)
            segs.append(torch.full(pad.shape, seg, dtype=torch.long, device=pad.device))
        x = torch.cat(xs, 1) + self.seg(torch.cat(segs, 1))
        att = (~torch.cat(pads, 1)).long()
        k = 0
        if self.bos is not None:                          # <s> в начале; срез выхода сдвигается на 1
            x = torch.cat([emb(torch.full((x.shape[0], 1), self.bos, device=x.device)).to(x.dtype), x], 1)
            att = torch.cat([torch.ones_like(att[:, :1]), att], 1)
            k = 1
        assert x.shape[1] <= self.limit, f"вход модели мира {x.shape[1]} > {self.limit} токенов"
        h = self.mlm.base_model(inputs_embeds=x, attention_mask=att).last_hidden_state[:, k:][:, out]
        return self.head_logits(h, self.ids)               # только по компактному словарю


def window_dist(logits, unimix):
    """Распределение токена окна: без <pad>/<mask>/<unk>/<sep> (их актор не увидел бы), с примесью равномерного
    (как unimix в DreamerV3). EMPTY разрешён: пустая ячейка."""
    logits = logits.float().clone()
    logits[..., :Vocab.EMPTY] = -1e9
    p = logits.softmax(-1)
    allowed = torch.ones(p.shape[-1], device=p.device)
    allowed[:Vocab.EMPTY] = 0
    return (1 - unimix) * p + unimix * allowed / allowed.sum()


def window_ce(logits, target):
    """CE окна по «сырым» логитам, без примеси unimix: когда вероятность нужного токена ниже примеси, градиент
    через смесь почти исчезает (так окно и застряло в прогоне pro_v2)."""
    logits = logits.float().masked_fill(torch.arange(logits.shape[-1], device=logits.device) < Vocab.EMPTY, -1e9)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def ste_embed(probs, hard, emb_weight, k):
    """STE: вперёд — эмбеддинг выбранного токена, назад — через смесь top-k токенов."""
    topv, topi = probs.topk(k, -1)
    topv = topv / topv.sum(-1, keepdim=True)
    soft = (topv.unsqueeze(-1).to(emb_weight.dtype) * emb_weight[topi]).sum(-2)
    return emb_weight[hard] + (soft - soft.detach())


class WorldModel(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        self.cfg = cfg
        L = cfg.max_state_tokens + 2 * cfg.window + cfg.max_action_tokens + 8
        if cfg.dry_run:                                   # заглушки: маленькие трансформеры с нуля
            make = lambda: Denoiser(vocab.size, cfg.wm_dim, cfg.wm_layers, cfg.wm_heads, L)
        else:                                             # предобученная BERT-подобная модель
            part = max(cfg.window, cfg.max_state_tokens, cfg.max_action_tokens) + 8
            make = lambda: MLMDenoiser(cfg.wm_backbone, vocab, cfg.wm_grad_ckpt, part)
        self.enc, self.dec, self.rssm = make(), make(), make()
        self.rm = ValueNet(vocab.size, cfg)               # RM: окно после шага → награда шага (метка — Shapley от VM)
        self.register_buffer("rm_sigma", torch.tensor(0.0))   # масштаб меток RM (задаётся по стартовым данным)
        if not cfg.dry_run:
            kind = "RoBERTa" if self.enc.roberta_head else "ModernBERT" if self.enc.modernbert_head else "общая"
            log(f"модель мира: 3 копии {cfg.wm_backbone}; голова {kind} по компактному словарю")
        self.main_steps = 0                               # шаги основной фазы (для нарастания веса KL)

    def set_reward_scale(self, frames):
        """Метки RM приводятся к единичному разбросу (иначе их потеря ничтожна рядом с остальными). Масштаб — по
        текущим меткам, после каждой игры VM: VM доучивается, и разброс её Shapley меняется."""
        phis = [st["phi"] for f in frames for st in f["steps"] if "phi" in st]
        self.rm_sigma.fill_(max(float(np.std(phis)) if len(phis) > 1 else 1.0, 1e-4))

    def reward(self, w):
        """Награда шага по окну после него — в единицах Shapley."""
        return self.rm(w).float() * self.rm_sigma

    def encode_logits(self, s_ids, s_pad):
        B, W = s_ids.shape[0], self.cfg.window
        win = torch.full((B, W), Vocab.MASK, dtype=torch.long, device=s_ids.device)
        nopad = torch.zeros((B, W), dtype=torch.bool, device=s_ids.device)
        return self.enc([(win, SEG_WIN_OUT, nopad), (s_ids, SEG_STATE, s_pad)], out=slice(0, W))   # окно — в начале

    @torch.no_grad()
    def predict(self, w_ids, a_ids, a_pad, steps, greedy=False):
        """RSSM: [l_t, a_t] → l_{t+1}: старт с l_t, несколько шагов правки (сначала самые уверенные позиции).
        greedy=True — без случайности (игра коалиций: разница значений — только от действия)."""
        B, W = w_ids.shape
        gen = torch.Generator(device=w_ids.device).manual_seed(0) if greedy else None   # игра коалиций: шум общий
        start_noise = torch.rand(w_ids.shape, generator=gen, device=w_ids.device) < self.cfg.rssm_start_noise
        cur = w_ids.masked_fill(start_noise, Vocab.MASK)  # старт — прошлое окно с долей простого шума (как в обучении)
        known = torch.zeros_like(w_ids, dtype=torch.bool)
        nopad = torch.zeros_like(w_ids, dtype=torch.bool)
        ar = torch.arange(W, device=w_ids.device).expand(B, W)
        for k in range(steps):
            with autocast():
                logits = self.rssm([(w_ids, SEG_WIN_IN, nopad), (cur, SEG_WIN_OUT, nopad), (a_ids, SEG_ACTION, a_pad)],
                                   out=slice(W, 2 * W))
            p = window_dist(logits, 0.0 if greedy else self.cfg.unimix)
            sample = p.argmax(-1) if greedy else torch.multinomial(p.view(-1, p.shape[-1]), 1).view(B, W)
            conf = p.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
            known = known | (sample == cur)               # совпадает с текущим — фиксируется сразу (это не правка)
            need = (math.ceil(W * (k + 1) / steps) - known.sum(1)).clamp(min=0)   # из правок — самые уверенные
            rank = torch.empty_like(ar).scatter_(1, conf.masked_fill(known, -1.0).argsort(1, descending=True), ar)
            commit = (rank < need[:, None]) & ~known
            cur = torch.where(commit, sample, cur)
            known = known | commit
        return cur

    def loss(self, s_ids, s_pad, a_ids, a_pad, n_ids, n_pad, copy_t=None, copy_n=None, phi=None, w_copy=0.0,
             rep_w=None, copy_only=False):
        cfg, W = self.cfg, self.cfg.window
        B = s_ids.shape[0]
        rep_w = cfg.rep_scale if rep_w is None else rep_w
        nopad = torch.zeros((B, W), dtype=torch.bool, device=s_ids.device)
        lt, ln = self.encode_logits(s_ids, s_pad), self.encode_logits(n_ids, n_pad)
        if copy_only:                                   # фаза копирования (предл.): учится только энкодер — 2 прохода
            copy_loss = (window_ce(lt, copy_t) + window_ce(ln, copy_n)) / 2
            return copy_loss, {"dec": 0.0, "kl": 0.0, "copy": copy_loss.item(), "rssm_acc": 0.0, "rm": 0.0}
        pt, pn = window_dist(lt, cfg.unimix), window_dist(ln, cfg.unimix)
        copy_loss = torch.zeros((), device=s_ids.device)
        if w_copy > 0 and copy_t is not None:           # копирование (предл.): окно = сжатый текст состояния
            copy_loss = (window_ce(lt, copy_t) + window_ce(ln, copy_n)) / 2
        V = pt.shape[-1]
        zt = torch.multinomial(pt.view(-1, V), 1).view(B, W)
        zn = torch.multinomial(pn.detach().view(-1, V), 1).view(B, W)
        # декодер: окно (STE) + зашумлённый s → s
        zt_dec = ste_embed(pt, zt, self.dec.emb_weight, cfg.topk_ste)
        dec_loss = 0.0
        for _ in range(cfg.dec_samples):              # ваше: несколько выборок, у каждой — случайная доля шума
            rho = cfg.dec_noise_lo + (cfg.dec_noise_hi - cfg.dec_noise_lo) * torch.rand((B, 1), device=s_ids.device)
            mask = (torch.rand(s_ids.shape, device=s_ids.device) < rho) & ~s_pad
            mask[:, 0] |= (mask.sum(1) == 0) & ~s_pad[:, 0]
            logits = self.dec([(zt_dec, SEG_WIN_IN, nopad), (s_ids.masked_fill(mask, Vocab.MASK), SEG_STATE, s_pad)],
                              out=slice(W, None))
            dec_loss = dec_loss + F.cross_entropy(logits[mask].float(), s_ids[mask])
        dec_loss = dec_loss / cfg.dec_samples
        # RSSM (ваше): [окно_t (STE), действие, x] → окно_{t+1}. Шум x — не случайные токены, а прошлое окно:
        # x стартует с l_t, часть позиций уже взята из l_{t+1} (промежуточные шаги восстановления), и ещё доля
        # простого шума [MASK], чтобы модель не халтурила (не просто переписывала l_t). Доли случайные; с каждой
        # пары — rssm_samples разных выборок. Потеря — по всем позициям окна.
        zt_rssm = ste_embed(pt, zt, self.rssm.emb_weight, cfg.topk_ste)
        pa = pn[..., Vocab.EMPTY:]
        dyn = rep = 0.0
        accs = []
        dev = s_ids.device
        for _ in range(cfg.rssm_samples):
            done = torch.rand((B, W), device=dev) < torch.rand((B, 1), device=dev)        # уже восстановлено
            noise = torch.rand((B, W), device=dev) < torch.rand((B, 1), device=dev)       # простой шум (у каждого x)
            x = torch.where(done, zn, zt).masked_fill(noise, Vocab.MASK)
            inf = ~(done & ~noise)                     # позиции, где ответа нет во входе (остальные — подсказка)
            inf[:, 0] |= ~inf.any(1)
            logits = self.rssm([(zt_rssm, SEG_WIN_IN, nopad), (x, SEG_WIN_OUT, nopad), (a_ids, SEG_ACTION, a_pad)],
                               out=slice(W, 2 * W))
            q = window_dist(logits, cfg.unimix)[..., Vocab.EMPTY:]              # служебные токены: вероятность 0
            kl_dyn = (pa.detach() * (pa.detach().log() - q.log())).sum(-1)       # учит RSSM
            kl_rep = (pa * (pa.log() - q.detach().log())).sum(-1)               # учит энкодер
            dyn = dyn + kl_dyn.clamp(min=cfg.free_nats)[inf].mean() / cfg.rssm_samples
            rep = rep + kl_rep.clamp(min=cfg.free_nats)[inf].mean() / cfg.rssm_samples
            accs.append((logits.argmax(-1) == pn.argmax(-1))[inf].float().mean())
        acc = torch.stack(accs).mean()
        # RM (ваше: учится вместе с моделью мира): окно следующего состояния (STE) → метка Shapley шага
        rm_loss = torch.zeros((), device=s_ids.device)
        if phi is not None and torch.isfinite(phi).any():
            ok = torch.isfinite(phi)
            zn_rm = ste_embed(pn[ok], zn[ok], self.rm.tok.weight, cfg.topk_ste)
            rm_loss = F.smooth_l1_loss(self.rm(zn_rm).float(), phi[ok] / self.rm_sigma.clamp(min=1e-4))
        total = dec_loss + cfg.dyn_scale * dyn + rep_w * rep + w_copy * copy_loss + cfg.rm_loss_scale * rm_loss
        return total, {"dec": dec_loss.item(), "kl": dyn.item(), "copy": copy_loss.item(), "rssm_acc": acc.item(),
                       "rm": rm_loss.item()}


class ValueNet(nn.Module):
    """Скаляр по окну: критик V(l) и RM r(l) (ваше: критик и RM работают от окна, как в Dreamer)."""
    def __init__(self, V, cfg):
        super().__init__()
        d = cfg.head_dim
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(cfg.window, d)
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, cfg.head_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, w):
        """w — id окна [B, W] или их эмбеддинги [B, W, d] (STE: тогда градиент идёт в энкодер)."""
        x = self.tok(w) if w.dtype == torch.long else w
        pos = torch.arange(x.shape[1], device=x.device)
        return self.head(self.body(x + self.pos(pos)[None]).mean(1)).squeeze(-1)


@torch.no_grad()
def encode_texts(wm, codec, texts, cfg, bs=128):
    """Окна для списка состояний (argmax энкодера, без градиента)."""
    wm.eval()
    out = []
    for s in range(0, len(texts), bs):
        ids, pad = pad_batch([codec.state_ids(t, cfg) for t in texts[s:s + bs]])
        with autocast():
            lg = wm.encode_logits(ids, pad)
        out.append(window_dist(lg.float(), 0.0).argmax(-1))
    return torch.cat(out, 0) if out else torch.zeros((0, cfg.window), dtype=torch.long, device=DEVICE)


# %% [markdown]
# ## 7. LLM: актор (он же исполнитель) и VM
# Актор — GPT + LoRA; исполнитель запросов «LLM» — та же модель с отключённым адаптером. KL считается к π0 (актор
# после обучения на своих удачных попытках; предл., вопр. 10). VM — BERT-подобная ModernBERT-base, целиком,
# с головой по среднему скрытых состояний. Удачи редки, поэтому в обучении VM у них больший вес; сырой логит VM
# сдвинут на log(вес), и vm_probs вычитает этот сдвиг — снаружи VM по-прежнему даёт P успеха (предл.).

# %%
def length_batches(lengths, max_n, max_tokens):
    """Индексы подсказок пачками для generate: по возрастанию длины; в пачке не больше max_n подсказок и не больше
    max_tokens токенов вместе с заполнением (длина × число). Меньше пустого заполнения; на коротких — пачки больше."""
    out, cur = [], []
    for i in sorted(range(len(lengths)), key=lambda j: lengths[j]):
        if cur and (len(cur) >= max_n or (len(cur) + 1) * lengths[i] > max_tokens):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + ([cur] if cur else [])


def left_pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, L - len(s):] = torch.tensor(s, dtype=torch.long)
        att[j, L - len(s):] = 1
    return ids.to(DEVICE), att.to(DEVICE)


class HFActor:
    def __init__(self, cfg, adapter=None, ref_adapter=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.actor_model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.text_tok = HFTok(self.tok)
        model = from_pretrained(AutoModelForCausalLM, cfg.actor_model).to(DEVICE)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            r = cfg.actor_lora_r
            model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                     "gate_proj", "up_proj", "down_proj"]))
        self.has_ref = bool(ref_adapter)
        if ref_adapter:
            model.load_adapter(ref_adapter, adapter_name="ref", is_trainable=False)
            model.set_adapter("default")
        self.model = model
        self.model.eval()
        g = model.generation_config                    # без сюрпризов от generation_config.json модели
        g.do_sample, g.temperature, g.top_p, g.top_k, g.repetition_penalty = False, None, None, None, 1.0
        self.mem_scale = 1.0                           # уменьшается, если пачка generate не влезла в память
        gen_eos = model.generation_config.eos_token_id
        self.eos = set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]) | {self.tok.eos_token_id}
        self.eos.discard(None)

    def prompt_ids(self, system, user, shots=None):
        msgs = [{"role": "system", "content": system}]
        for u, a in shots or []:
            msgs += [{"role": "user", "content": "STATE:\n" + u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False).input_ids

    @contextlib.contextmanager
    def adapter(self, which):
        """default — обучаемый адаптер; ref — π0 (если загружен); base — без адаптера."""
        if which == "ref" and self.has_ref:
            self.model.set_adapter("ref")
            try:
                yield
            finally:
                self.model.set_adapter("default")
        elif which in ("ref", "base"):
            with self.model.disable_adapter():
                yield
        else:
            yield

    def cut_stops(self, row):
        """Обрезать ответ на первой стоп-строке (модель начала писать следующее состояние) и закрыть его EOS."""
        text = self.tok.decode(row, skip_special_tokens=True)
        hits = [i for i in (text.find(x) for x in STOP_STRINGS) if i >= 0]
        if not hits:
            return row, text
        idx, keep, pos = min(hits), [], 0
        for t, pc in zip(row, self.pieces(row)):
            if pos + len(pc) > idx:
                break
            keep.append(t)
            pos += len(pc)
        keep.append(self.tok.eos_token_id)
        return keep, self.tok.decode(keep, skip_special_tokens=True)

    def pieces(self, ids):
        return [self.tok.decode([i], skip_special_tokens=True) for i in ids]

    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @torch.no_grad()
    def generate(self, prompts, max_new, greedy, temperature=1.0, which="default"):
        from transformers import GenerationConfig
        self.model.eval()
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(self.eos),
                  stop_strings=STOP_STRINGS, use_cache=True)
        gc = GenerationConfig(do_sample=False, **kw) if greedy else \
            GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, **kw)
        outs = [None] * len(prompts)
        queue = length_batches([len(p) + max_new for p in prompts], self.cfg.gen_batch,
                               int(self.cfg.gen_tokens * self.mem_scale))
        with self.adapter(which):
            while queue:
                idx = queue.pop(0)
                ids, att = left_pad([prompts[i] for i in idx], self.tok.pad_token_id)
                oom = False
                try:
                    try:
                        gen = self.model.generate(input_ids=ids, attention_mask=att, generation_config=gc,
                                                  tokenizer=self.tok)
                    except (TypeError, ValueError) as e:     # версия без stop_strings: обрежет cut_stops ниже
                        if gc.stop_strings is None:
                            raise
                        log("ВНИМАНИЕ: generate() не принял stop_strings (", repr(e)[:160], ") — генерирую без них")
                        gc.stop_strings = None
                        gen = self.model.generate(input_ids=ids, attention_mask=att, generation_config=gc)
                except torch.cuda.OutOfMemoryError:
                    if len(idx) == 1:
                        raise
                    oom = True
                if oom:                                      # память освобождается только после выхода из except
                    del ids, att
                    free_gpu()
                    used = len(idx) * max(len(prompts[i]) + max_new for i in idx)
                    self.mem_scale = min(self.mem_scale, used / 2 / self.cfg.gen_tokens)
                    rest = idx + [i for b in queue for i in b]   # всё оставшееся — заново пачками по новому бюджету
                    queue = [[rest[j] for j in b] for b in length_batches(
                        [len(prompts[i]) + max_new for i in rest], self.cfg.gen_batch,
                        int(self.cfg.gen_tokens * self.mem_scale))]
                    log(f"ВНИМАНИЕ: не хватило памяти GPU на пачку из {len(idx)} подсказок — дальше пачки вдвое меньше "
                        f"(бюджет ×{self.mem_scale:.2f})")
                    continue
                for i, row in zip(idx, gen[:, ids.shape[1]:].tolist()):
                    cut = next((k for k, t in enumerate(row) if t in self.eos), None)
                    row = row[:cut + 1] if cut is not None else row
                    if not row:
                        row = [self.tok.eos_token_id]
                    outs[i] = self.cut_stops(row)
        return outs

    def execute(self, requests):
        """Исполнитель запросов «LLM»: базовая модель без адаптера, жадно."""
        prompts = [self.prompt_ids(TASK.system_exec, r) for r in requests]
        outs = self.generate(prompts, TASK.exec_max_new, greedy=True, which="base")
        return [TASK.norm_result(t) for _, t in outs]

    def seq_logprobs(self, prompts, gens, which="default"):
        """log π каждого сгенерированного токена. which: default (с градиентом, если включён) / ref / base."""
        out = []
        with self.adapter(which):
            seqs = [p + g for p, g in zip(prompts, gens)]
            ids, att = left_pad(seqs, self.tok.pad_token_id)
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            K = max(len(g) for g in gens)
            try:
                logits = self.model(input_ids=ids, attention_mask=att, position_ids=pos,
                                    logits_to_keep=K + 1, use_cache=False).logits
            except TypeError:
                logits = self.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False).logits
            logits = logits[:, -(K + 1):-1]
            for j, g in enumerate(gens):
                n = len(g)
                lp = logits[j, K - n:K].float().log_softmax(-1)
                out.append(lp.gather(-1, torch.tensor(g, device=lp.device).unsqueeze(-1)).squeeze(-1))
        return out

    def train_mode(self, flag):
        self.model.train(flag)

    def save_adapter(self, path):
        self.model.save_pretrained(path, selected_adapters=["default"])

    def encode_action(self, text):
        """Токены действия (как после generate: текст + EOS) — для π0 по вычищенному действию."""
        return self.text_tok.encode(text) + [self.tok.eos_token_id]


class StubActor:
    """Заглушка актора (DRY_RUN): разумные случайные действия по тексту и крошечный обучаемый «словарь»."""
    def __init__(self, cfg, adapter=None, ref_adapter=None):
        self.cfg = cfg
        self.text_tok = CharTok()
        self.tok = self.text_tok
        self.w = nn.Parameter(torch.zeros(len(self.text_tok.itos), device=DEVICE))
        if adapter and os.path.exists(os.path.join(adapter, "stub.pt")):
            self.w.data.copy_(torch.load(os.path.join(adapter, "stub.pt")))
        self.base = torch.zeros_like(self.w.data)        # «без адаптера»
        self.ref = self.base
        if ref_adapter and os.path.exists(os.path.join(ref_adapter, "stub.pt")):
            self.ref = torch.load(os.path.join(ref_adapter, "stub.pt")).to(DEVICE)
        self.has_ref = bool(ref_adapter)
        self.eos = {self.text_tok.eos_token_id}
        self.rng = random.Random(0)

    def prompt_ids(self, system, user, shots=None):
        return self.text_tok.encode(user)

    def pieces(self, ids):
        return [self.text_tok.itos[i] if i > 2 else "" for i in ids]

    def params(self):
        return [self.w]

    @torch.no_grad()
    def generate(self, prompts, max_new, greedy, temperature=1.0, which="default"):
        outs = []
        for p in prompts:
            text = TASK.stub_action(self.text_tok.decode(p), self.rng, self.cfg)[:max_new]
            outs.append((self.text_tok.encode(text) + [self.text_tok.eos_token_id], text))
        return outs

    def execute(self, requests):
        return [TASK.stub_exec(r, self.rng) for r in requests]

    def seq_logprobs(self, prompts, gens, which="default"):
        w = self.w if which == "default" else self.ref if which == "ref" and self.has_ref else self.base
        lp = w.log_softmax(-1)
        return [lp[torch.tensor(g, device=DEVICE)] for g in gens]

    def train_mode(self, flag):
        pass

    def save_adapter(self, path):
        os.makedirs(path, exist_ok=True)
        save_torch(self.w.detach().cpu(), os.path.join(path, "stub.pt"))

    def encode_action(self, text):
        return self.text_tok.encode(text) + [self.text_tok.eos_token_id]


def make_actor(cfg, adapter=None, ref_adapter=None):
    return StubActor(cfg, adapter, ref_adapter) if cfg.dry_run else HFActor(cfg, adapter, ref_adapter)


BERT_TYPES = ("modernbert", "bert", "roberta", "distilbert", "deberta", "deberta-v2", "electra", "xlm-roberta")


def load_encoder(name, path=None):
    """BERT-подобная модель (AutoModel) в fp32: без torch.compile внутри (ModernBERT), внимание sdpa."""
    from transformers import AutoConfig, AutoModel
    src = path or name
    mc = AutoConfig.from_pretrained(src)
    if hasattr(mc, "reference_compile"):
        mc.reference_compile = False
    err = None
    for kw in ({"dtype": torch.float32, "attn_implementation": "sdpa"}, {"torch_dtype": torch.float32}, {}):
        try:
            return AutoModel.from_pretrained(src, config=mc, **kw).float()
        except (TypeError, ValueError) as e:
            err = e
    raise err


class HFVM(nn.Module):
    """VM: запрос + путь + траектория → логит пользы (P успеха). BERT-подобная модель (по умолчанию ModernBERT-base):
    целиком дообучается, вход двунаправленный, оценка — по среднему скрытых состояний. Декодер (например, Qwen) —
    с LoRA и оценкой по последнему токену."""
    def __init__(self, cfg, adapter=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.vm_model)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.bert = AutoConfig.from_pretrained(cfg.vm_model).model_type in BERT_TYPES
        if self.bert:
            saved = adapter and os.path.exists(os.path.join(adapter, "config.json"))
            enc = load_encoder(cfg.vm_model, adapter if saved else None)
            if cfg.vm_grad_ckpt:
                try:
                    enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    enc.gradient_checkpointing_enable()
        else:
            from peft import LoraConfig, PeftModel, get_peft_model
            enc = from_pretrained(AutoModel, cfg.vm_model)
            if adapter:
                enc = PeftModel.from_pretrained(enc, adapter, is_trainable=True)
            else:
                r = cfg.vm_lora_r
                enc = get_peft_model(enc, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                                     target_modules=["q_proj", "v_proj"]))
        self.enc = enc
        self.head = nn.Linear(enc.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if adapter and os.path.exists(os.path.join(adapter, "head.pt")):
            self.head.load_state_dict(torch.load(os.path.join(adapter, "head.pt")))
        self.logit_offset = load_vm_offset(adapter)
        self.to(DEVICE)

    def forward(self, texts):
        M, H = self.cfg.vm_max_tokens, 48                 # длинная траектория: запрос (начало) + последние шаги
        rows = [x if len(x) <= M else x[:H] + x[-(M - H):] for x in self.tok(texts, add_special_tokens=False).input_ids]
        if self.bert:                                     # [CLS] … [SEP], как при предобучении
            cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
            rows = [([cls] if cls is not None else []) + x + ([sep] if sep is not None else []) for x in rows]
        L = max(len(x) for x in rows)
        ids = torch.full((len(rows), L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for i, x in enumerate(rows):
            ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            att[i, :len(x)] = 1
        ids, att = ids.to(DEVICE), att.to(DEVICE)
        with autocast():
            hid = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        if self.bert:                                     # среднее по настоящим токенам
            pooled = (hid.float() * att[..., None]).sum(1) / att.sum(1, keepdim=True)
        else:                                             # последний токен
            pooled = hid[torch.arange(len(texts), device=DEVICE), att.sum(-1) - 1].float()
        return self.head(F.layer_norm(pooled, pooled.shape[-1:])).squeeze(-1)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.bert:                                     # целиком, в bf16 (вдвое меньше; load_encoder вернёт fp32)
            sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in self.enc.state_dict().items()}
            try:
                self.enc.save_pretrained(path, state_dict=sd)
            except TypeError:                             # версия без state_dict= : как есть, в fp32
                self.enc.save_pretrained(path)
        else:
            self.enc.save_pretrained(path)                # декодер — только LoRA
        save_torch(self.head.state_dict(), os.path.join(path, "head.pt"))
        save_json({"logit_offset": self.logit_offset}, os.path.join(path, "vm_meta.json"))


class StubVM(nn.Module):
    """Заглушка VM (DRY_RUN): признаки по хешам символьных триграмм → логит."""
    def __init__(self, cfg, adapter=None):
        super().__init__()
        self.cfg = cfg
        self.lin = nn.Linear(512, 1)
        if adapter and os.path.exists(os.path.join(adapter, "stub_vm.pt")):
            self.lin.load_state_dict(torch.load(os.path.join(adapter, "stub_vm.pt")))
        self.logit_offset = load_vm_offset(adapter)
        self.to(DEVICE)

    def forward(self, texts):
        x = torch.zeros((len(texts), 512), device=DEVICE)
        for i, t in enumerate(texts):
            idx = [zlib.crc32(t[j:j + 3].encode()) % 512 for j in range(len(t) - 2)]
            x[i] = torch.bincount(torch.tensor(idx, dtype=torch.long), minlength=512).float().to(DEVICE)
        return self.lin(F.normalize(x, dim=-1)).squeeze(-1)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_torch(self.lin.state_dict(), os.path.join(path, "stub_vm.pt"))
        save_json({"logit_offset": self.logit_offset}, os.path.join(path, "vm_meta.json"))


def load_vm_offset(adapter):
    """Сдвиги логита VM (верх, подзадачи) — подобраны после обучения (см. fit_vm_offsets); хранятся рядом с весами."""
    meta = load_json(os.path.join(adapter, "vm_meta.json")) if adapter else None
    x = (meta or {}).get("logit_offset", 0.0)
    return [float(v) for v in x] if isinstance(x, list) else [float(x), float(x)]


def is_sub_text(text):
    """Вход VM подзадачи (у неё есть путь сверху) — у верхнего уровня пути нет."""
    return "\nPATH (context):" in text


def vm_offset_of(vm, text):
    return vm.logit_offset[1] if is_sub_text(text) else vm.logit_offset[0]


def make_vm(cfg, adapter=None):
    return StubVM(cfg, adapter) if cfg.dry_run else HFVM(cfg, adapter)


# %% [markdown]
# ## 8. Настоящее исполнение с рекурсией
# Все уровни всех задач идут пачками: на каждом такте актор действует во всех готовых кадрах сразу, запросы «LLM»
# исполняются одной пачкой, запросы «SUB» открывают кадры уровнем ниже (та же модель).
# В режиме window актор видит только окно (ваше, W7); в режиме text — полный текст состояния (датасет и GRPO).

# %%
def new_frame(frames, request, level, parent, truth, root_query, path=(), subs=0):
    fr = {"id": len(frames), "request": request, "level": level, "parent": parent, "plan": None, "steps": [],
          "answer": None, "done": False, "pending": 0, "truth": truth, "root_query": root_query, "path": list(path),
          "subs_left": subs}
    frames.append(fr)
    return fr


def finish(frames, fr, answer):
    fr["done"], fr["answer"] = True, answer
    if fr["parent"] is not None:
        pid, si, j = fr["parent"]
        deliver(frames, frames[pid], si, j, answer if answer is not None else "NO ANSWER")


def deliver(frames, fr, si, j, result):
    fr["steps"][si]["reqs"][j]["result"] = str(result)[:max(40, CFG.max_answer_chars)]
    fr["pending"] -= 1
    if fr["pending"] == 0 and len(fr["steps"]) >= CFG.max_steps:
        finish(frames, fr, None)


def run_episodes(tasks, actor, cfg, mode, wm=None, codec=None, greedy=False, few_shot=False, which="default"):
    frames = []
    tops = [new_frame(frames, t["query"], 0, None, t["truth"], t["query"], subs=sub_allow(cfg, 0)) for t in tasks]
    for f, t in zip(tops, tasks):
        f["ppl"] = t.get("ppl")
        for k in ("task_id", "attempt"):                 # датасет: номер задачи и попытки (пары для VM)
            if k in t:
                f[k] = t[k]
    root_of = {f["id"]: f["id"] for f in tops}         # общий счётчик: подзадач на одну задачу верхнего уровня
    size = {f["id"]: 0 for f in tops}
    pr = Progress("эпизоды", len(tops))
    for _ in range((cfg.sub_total + 2) * cfg.max_steps * 2 + 8):   # худшая цепочка в пределах счётчика
        ready = [f for f in frames if not f["done"] and f["pending"] == 0]
        if not ready:
            break
        pr.tick(sum(f["done"] for f in tops), f" (в работе кадров {sum(not f['done'] for f in frames)}, "
                                              f"всего {len(frames)}, глубже всех — уровень {max(f['level'] for f in frames)})")
        states = [render_state(f["request"], f["level"], f["plan"], f["steps"], cfg, f["path"], f["subs_left"])
                  for f in ready]
        allow = [f["subs_left"] > 0 and size[root_of[f["id"]]] < cfg.sub_total for f in ready]
        if mode == "window":
            wins = encode_texts(wm, codec, states, cfg)
            contents = [codec.window_text(w.tolist()) for w in wins]
        else:
            contents = states
        prompts = [actor_prompt(actor, cfg, c, few_shot, a) for c, a in zip(contents, allow)]
        acts = actor.generate(prompts, cfg.gen_action_tokens, greedy, cfg.temperature, which=which)
        jobs = []
        for f, s_text, c, p, (gids, text), al in zip(ready, states, contents, prompts, acts, allow):
            plan, reqs, ans, _ = parse_action(text, f["level"], cfg)
            f["steps"].append({"state": s_text, "prompt": p, "gen": gids, "action": text, "plan": plan, "reqs": reqs,
                               "answer": ans, "subs_before": f["subs_left"], "allow_sub": al,
                               **({"window": c} if mode == "window" else {})})   # что видел актор (для анализа)
            if plan:
                f["plan"] = plan
            si = len(f["steps"]) - 1
            if ans is not None:
                finish(frames, f, ans)
            elif not reqs:
                if len(f["steps"]) >= cfg.max_steps:
                    finish(frames, f, None)
            else:
                f["pending"] = len(reqs)
                for j, r in enumerate(reqs):
                    root = root_of[f["id"]]
                    if r["dest"] == "SUB" and (f["subs_left"] <= 0 or size[root] >= cfg.sub_total):
                        r["dest"] = "LLM"                  # счётчик кончился: подзадача — обычным запросом

                    if r["dest"] == "SUB":                  # подзадача получает весь путь к этому шагу (ваше)
                        ch = new_frame(frames, r["text"], f["level"] + 1, (f["id"], si, j),
                                       TASK.sub_truth(r["text"]), f["root_query"], context_block(f, si),
                                       sub_allow(cfg, f["level"] + 1))
                        r["child"] = ch["id"]
                        root_of[ch["id"]] = root
                        size[root] += 1
                        f["subs_left"] -= 1
                    else:                                  # запрос LLM — тоже со всем путём к шагу (ваше)
                        jobs.append((f, si, j, exec_content(f, si, r["text"])))
        if jobs:
            for (f, si, j, _), res in zip(jobs, actor.execute([j[3] for j in jobs])):
                deliver(frames, f, si, j, strip_echo(f["steps"][si]["reqs"][j]["text"], res))
    for f in frames:                                   # страховка: незавершённые закрываются без ответа
        if not f["done"]:
            f["done"] = True
    for f in tops:
        f["R"] = float(check_answer(f["answer"], f["truth"]))
    byid = {f["id"]: f for f in frames}
    for f in frames:
        for t, st in enumerate(f["steps"]):
            st["next_state"] = (f["steps"][t + 1]["state"] if t + 1 < len(f["steps"])
                                else render_state(f["request"], f["level"], f["plan"], f["steps"], cfg, f["path"],
                                                  f["subs_left"]))
        root = f
        while root["parent"] is not None:
            root = byid[root["parent"][0]]
        f["root_R"] = root.get("R", 0.0)
        if f["level"] > 0:                             # только диагностика: настоящий ответ подзадачи
            f["sub_ok"] = None if f["truth"] is None else bool(check_answer(f["answer"], f["truth"]))
    return tops, frames


def episode_stats(tops, frames):
    llm = sum(1 for f in frames for st in f["steps"] for r in st["reqs"] if r["dest"] == "LLM")
    sub = sum(1 for f in frames for st in f["steps"] for r in st["reqs"] if r["dest"] == "SUB")
    by_ppl = {}
    for f in tops:
        if f.get("ppl") is not None:
            by_ppl.setdefault(str(f["ppl"]), []).append(f["R"])
    return {"acc": float(np.mean([f["R"] for f in tops])) if tops else 0.0,
            **({"by_ppl": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_ppl.items())}} if by_ppl else {}),
            "answered": float(np.mean([f["answer"] is not None for f in tops])) if tops else 0.0,
            "llm_calls": llm / max(1, len(tops)), "sub_calls": sub / max(1, len(tops)),
            "steps_top": float(np.mean([len(f["steps"]) for f in tops])) if tops else 0.0}


def merge_frames(parts):
    """Склеить кадры нескольких прогонов: id — позиция в общем списке, ссылки на родителя сдвигаются."""
    out = []
    for part in parts:
        off = len(out)
        for f in part:
            g = dict(f, id=f["id"] + off)
            if g["parent"] is not None:
                g["parent"] = [g["parent"][0] + off] + list(g["parent"][1:])
            g["steps"] = [dict(st, reqs=[dict(r, child=r["child"] + off) if "child" in r else r for r in st["reqs"]])
                          for st in f["steps"]]
            out.append(g)
    return out


def strip_frames(frames, drop=("prompt",)):
    """Для сохранения на Drive: без подсказок (их легко пересобрать) и без лишнего (drop — поля шагов; токены
    действия gen нужны только датасету — для π0)."""
    out = []
    for f in frames:
        g = {k: v for k, v in f.items() if k != "steps"}
        g["steps"] = [{k: v for k, v in st.items() if k not in drop} for st in f["steps"]]
        out.append(g)
    return out


# %% [markdown]
# ## 9. VM и метки RM
# * VM учится наверху по R; для подзадач — только слабая метка «верно», если задача наверху решена (предл.);
#   награда подзадачи — её собственная оценка VM (ваше, W8). Настоящие ответы подзадач — только для диагностики.
# * Игра VM: игроки — шаги уровня (действия шага вместе с результатами); коалиция — траектория без остальных шагов.
#   Доля Shapley шага — метка RM для окна после этого шага (ваше, W3–W4).

# %%
def shapley(m, v):
    """Точный Shapley; v — словарь {отсортированный кортеж игроков: значение}."""
    phi = [0.0] * m
    for i in range(m):
        others = [j for j in range(m) if j != i]
        for r in range(m):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
                phi[i] += w * (v[tuple(sorted(S + (i,)))] - v[S])
    return phi


def vm_examples(frames, cfg, levels=False):
    """Примеры VM: наверху метка — R. У подзадачи (предл.) — исход задачи наверху (1 — решена, 0 — нет), без ответа
    — 0; вес sub_weight. Раньше подзадачи с ответом при нерешённой задаче в обучение не шли, и у подзадач метка
    совпадала с «был ли ответ» (VM учила «ответила — значит полезно», а не «решено»). levels=True — ещё флаг
    «верхний уровень» (для диагностики)."""
    ex = []
    for f in frames:
        if not f["steps"]:
            continue
        top = f["level"] == 0
        r = f.get("R", 0.0) if top else (f.get("root_R", 0.0) if f["answer"] is not None else 0.0)
        y, w = (1.0 if r > 0.5 else 0.0), (1.0 if top else cfg.sub_weight)
        ex.append((vm_text(f), y, w, top) if levels else (vm_text(f), y, w))
    return ex


def vm_point_data(examples, cfg, budget=None):
    """Поточечные примеры VM (предл.): удач ~2 %, поэтому их вес w = N/P (P, N — суммарный вес удач и неудач), но не
    больше vm_pos_weight_max. budget — сколько примеров взять: все удачи + случайные неудачи с весом 1/q (q — доля
    взятых), чтобы выборка не меняла баланс. Сырой логит тогда смещён на log w; vm_probs вычитает сдвиг — снаружи
    VM даёт P успеха (сдвиги потом подбираются по уровням: fit_vm_offsets). → (данные, w, средний вес, число удач)."""
    pos = [(t, y, w) for t, y, w in examples if y > 0.5]
    neg = [(t, y, w) for t, y, w in examples if y <= 0.5]
    if not pos and not neg:
        return [], 1.0, 0.0, 0
    P, N = sum(w for _, _, w in pos), sum(w for _, _, w in neg)
    cap = cfg.vm_pos_weight_max
    wpos = min(cap, max(1.0 / cap, N / P)) if P > 0 and N > 0 else 1.0
    k = len(neg) if budget is None else min(len(neg), max(budget - len(pos), budget // 2))
    q = k / len(neg) if neg else 1.0
    data = [(t, y, w * wpos) for t, y, w in pos] + [(t, y, w / q) for t, y, w in random.sample(neg, k)]
    wbar = sum(e[2] for e in data) / max(1, len(data))   # постоянный делитель: делить на сумму весов своей пачки
    return data, wpos, wbar, len(pos)                     # нельзя — удача раздувает её сама, и вес удач выходит не w


def vm_optimizer(vm, cfg):
    heads = [p for n, p in vm.named_parameters() if p.requires_grad and n.startswith(("head", "lin"))]
    rest = [p for n, p in vm.named_parameters() if p.requires_grad and not n.startswith(("head", "lin"))]
    groups = [{"params": heads, "lr": 1e-3}] + ([{"params": rest, "lr": cfg.vm_lr}] if rest else [])
    return torch.optim.AdamW(groups), [p for g in groups for p in g["params"]]


@torch.no_grad()
def fit_vm_offsets(vm, data, wpos):
    """Сдвиги логита по уровням, подобранные после обучения (предл.): средняя P успеха на обучающих примерах (с весами
    совокупности: удачи без ×wpos, неудачи с 1/q) равна доле удач — отдельно наверху и у подзадач. Одного сдвига log w
    мало: ограничение нормы градиента срабатывает почти на каждой пачке с удачей и гасит её вес — по уровням по-разному."""
    vm.eval()
    raw = []
    for s in range(0, len(data), 32):
        raw += vm([e[0] for e in data[s:s + 32]]).float().tolist()
    raw = np.array(raw, dtype=np.float64)
    w0 = np.array([e[2] / wpos if e[1] > 0.5 else e[2] for e in data], dtype=np.float64)
    y = np.array([e[1] for e in data], dtype=np.float64)
    sub = np.array([is_sub_text(e[0]) for e in data], dtype=bool)
    out = []
    for m in (~sub, sub):
        t = float((w0[m] * y[m]).sum())
        if not m.any() or t <= 0 or t >= float(w0[m].sum()):
            out.append(round(math.log(wpos), 4))       # на уровне нет удач (или неудач): подбирать не по чему
            continue
        lo, hi = -40.0, 40.0
        for _ in range(60):                            # средняя P монотонно падает с ростом сдвига — делим пополам
            c = (lo + hi) / 2
            lo, hi = (c, hi) if float((w0[m] / (1 + np.exp(-(raw[m] - c)))).sum()) > t else (lo, c)
        out.append(round((lo + hi) / 2, 4))
    return out


def vm_point_loss(vm, b, wbar):
    logits = vm([e[0] for e in b])
    y = torch.tensor([e[1] for e in b], device=DEVICE)
    w = torch.tensor([e[2] for e in b], device=DEVICE)
    return (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / (len(b) * wbar)


def train_vm(vm, examples, cfg, budget=None):
    """VM поточечно: взвешенная бинарная кросс-энтропия (см. vm_point_data)."""
    data, wpos, wbar, npos = vm_point_data(examples, cfg, budget)
    if not data or wbar <= 0:
        return {}
    opt, params = vm_optimizer(vm, cfg)
    vm.train()
    tot, n = 0.0, 0
    for _ in range(cfg.vm_epochs):
        random.shuffle(data)
        for s in range(0, len(data), cfg.vm_batch):
            b = data[s:s + cfg.vm_batch]
            loss = vm_point_loss(vm, b, wbar)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(b), n + len(b)
    vm.logit_offset = fit_vm_offsets(vm, data, wpos)
    return {"vm_loss": tot / max(1, n), "vm_pos": npos, "vm_n": len(data), "vm_w": round(wpos, 2),
            "vm_offset": vm.logit_offset}


def train_vm_pairs(vm, pairs, examples, cfg, budget=None, epochs=1):
    """VM попарно (ваше; Брэдли–Терри, как модель награды в SCAR): −log σ(VM(x, y+) − VM(x, y−)), вес пары —
    уверенность судьи (по R — 1). Плюс поточечная часть с весом vm_point_weight (честные метки, вес удач; предл.): она
    держит шкалу — снаружи VM по-прежнему даёт P успеха — и учит VM входам подзадач: пары есть только наверху, а игра
    Shapley идёт на всех уровнях. Без пар (vm_pairs = none или пар нет) — только поточечно."""
    if not pairs or cfg.vm_pairs == "none":
        return train_vm(vm, examples, cfg, budget)
    data, wpos, wbar, npos = vm_point_data(examples, cfg, max(cfg.vm_iter_examples, len(pairs))
                                           if budget is None else budget)
    pw = sum(x[2] for x in pairs) / len(pairs)
    opt, params = vm_optimizer(vm, cfg)
    vm.train()
    B = max(1, cfg.vm_batch // 2)                         # пар на шаг (две текста на пару) + столько же поточечных
    steps = max(math.ceil(len(pairs) / B), math.ceil(len(data) / B) if data and wbar > 0 else 0)
    tot_bt = tot_pt = acc = 0.0
    n, dd = 0, []
    for _ in range(epochs):
        pp = random.sample(pairs, len(pairs))
        dd = random.sample(data, len(data)) if data and wbar > 0 else []
        for s in range(steps):
            b = [pp[(s * B + i) % len(pp)] for i in range(min(B, len(pp)))]
            opt.zero_grad()
            sc = vm([x[0] for x in b] + [x[1] for x in b]).float()
            d = sc[:len(b)] - sc[len(b):]
            w = torch.tensor([x[2] for x in b], device=DEVICE)
            loss = (F.softplus(-d) * w).sum() / (len(b) * pw)
            loss.backward()                                 # попарная и поточечная части — отдельными проходами:
            tot_bt += loss.item()                           # пик памяти — от одной, а не от суммы
            acc += (d > 0).float().mean().item()
            del sc, d, loss
            if dd and cfg.vm_point_weight > 0:
                lp = vm_point_loss(vm, [dd[(s * B + i) % len(dd)] for i in range(min(B, len(dd)))], wbar)
                (cfg.vm_point_weight * lp).backward()
                tot_pt += lp.item()
                del lp
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            n += 1
    vm.eval()
    if dd:
        vm.logit_offset = fit_vm_offsets(vm, data, wpos)
    return {"vm_pair_loss": round(tot_bt / max(1, n), 4), "vm_point_loss": round(tot_pt / max(1, n), 4),
            "vm_pair_train_acc": round(acc / max(1, n), 3), "vm_pairs": len(pairs), "vm_pos": npos, "vm_n": len(data),
            "vm_w": round(wpos, 2), "vm_offset": vm.logit_offset}


def held_task(request):
    """Отложенная задача обучения (10 % по хешу условия): её пары — только для проверки VM, её кадры не идут в
    поточечное обучение VM (иначе VM запомнит решение и проверка завысится)."""
    return zlib.crc32(request.encode()) % 10 == 0


def split_pairs(pairs):
    """Пары для обучения и отложенные (10 % задач — по группе пары): на них — доля верно упорядоченных пар."""
    return [x for x in pairs if x[3] % 10], [x for x in pairs if x[3] % 10 == 0]


def vm_pair_acc(vm, pairs, limit=600):
    """Доля пар, где VM ставит победителя выше (0.5 — наугад); None — пар нет."""
    pairs = pairs[:limit]
    if not pairs:
        return None
    p = vm_probs(vm, [x[0] for x in pairs] + [x[1] for x in pairs])
    n = len(pairs)
    return round(float(np.mean([(u > v) + 0.5 * (u == v) for u, v in zip(p[:n], p[n:])])), 3)


@torch.no_grad()
def vm_probs(vm, texts, bs=32):
    """P успеха: сигмоида от сырого логита VM минус сдвиг своего уровня (верх / подзадачи)."""
    vm.eval()
    out, pr = [], Progress("VM", len(texts))
    for s in range(0, len(texts), bs):
        off = torch.tensor([vm_offset_of(vm, t) for t in texts[s:s + bs]], device=DEVICE)
        out += torch.sigmoid(vm(texts[s:s + bs]).float() - off).tolist()
        pr.tick(len(out))
    return out


def auc(scores, labels):
    """Доля пар (удача, неудача), где удача оценена выше (0.5 — наугад); None, если нет одного из классов."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    p, q = s[y], s[~y]
    if not len(p) or not len(q) or not np.isfinite(s).all():
        return None
    return float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean())


def vm_game(vm, frames, cfg, refresh=1.0):
    """Shapley по шагам уровня в игре VM → step['phi'] (метка RM). refresh < 1 (предл., ради времени): метки
    пересчитываются у всех новых кадров и у такой доли старых; остальные старые хранят метки прошлой VM."""
    jobs, texts = [], []
    for f in frames:
        m = len(f["steps"])
        if m == 0:
            continue
        if refresh < 1.0 and all("phi" in st for st in f["steps"]) and random.random() > refresh:
            continue
        subsets = [S for r in range(m + 1) for S in itertools.combinations(range(m), r)]
        jobs.append((f, m, subsets, len(texts)))
        texts += [vm_text(f, set(S)) for S in subsets]
    probs = vm_probs(vm, texts)
    phis, by = [], {}
    for f, m, subsets, off in jobs:
        v = {S: probs[off + k] for k, S in enumerate(subsets)}
        for st, ph in zip(f["steps"], shapley(m, v)):
            st["phi"] = ph
            phis.append(ph)
            kind = "answer" if st.get("answer") is not None else ("requests" if st.get("reqs") else "empty")
            by.setdefault(f"phi_{'top' if f['level'] == 0 else 'sub'}_{kind}", []).append(ph)
    return {"phi_mean": float(np.mean(phis)) if phis else 0.0, "phi_abs": float(np.mean(np.abs(phis))) if phis else 0.0,
            "vm_texts": len(texts), **{k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())},
            **{"n_" + k[4:]: len(v) for k, v in sorted(by.items())}}


def vm_diagnostics(vm, frames):
    """Качество VM: наверху — против R; на подзадачах — против настоящего ответа (только диагностика).
    vm_auc — по полному тексту: его почти целиком объясняет сложность задачи (число жителей в запросе), а она
    одинакова во всех коалициях и заслуги шагам не даёт. vm_gain_auc — по v(вся траектория) − v(без шагов), т. е. по
    сумме меток Shapley кадра: 0.5 у VM, которая шагов не различает (предл.)."""
    out = {}
    for name, sel in (("top", lambda f: f["level"] == 0), ("sub", lambda f: f["level"] > 0 and f.get("sub_ok") is not None)):
        fs = [f for f in frames if sel(f) and f["steps"]]
        if fs:
            s = vm_probs(vm, [vm_text(f) for f in fs])
            s0 = vm_probs(vm, [vm_text(f, set()) for f in fs])
            lab = [f["R"] > 0.5 if f["level"] == 0 else f["sub_ok"] for f in fs]
            out[f"vm_auc_{name}"] = auc(s, lab)
            out[f"vm_gain_auc_{name}"] = auc([a - b for a, b in zip(s, s0)], lab)   # заслуга шагов (предл.)
            out[f"n_{name}"] = len(fs)
    return out


# %% [markdown]
# ## 9a. Судья пар для VM (ваше: VM учится попарно, как модель награды в SCAR; судья — сильная модель с CoT)
# * Пара — две попытки одной задачи (верхний уровень). Исходы разные — метка из R, бесплатно и точно. Исходы
#   одинаковые (почти всегда обе неудачи) — судит модель через OpenRouter: видит условие, эталон, объяснение формата
#   и шаги обеих попыток — ровно то, что потом оценивает VM (предл.). Порядок A/B — случайный; на доле пар — оба
#   порядка (смещение по позиции). Часть пар с разным исходом тоже идёт судье — проверка согласия с R.
# * Ключ — только из переменной окружения или Colab Secrets (OPENROUTER_API_KEY): не в Config (Config целиком пишется
#   в log.txt), не в лог, не на Drive. Ответы — в кэш на Drive; предел расхода — по usage из ответов API.

# %%
JUDGE_SYSTEM = """You are a careful judge of solutions to "Knights and Knaves" logic puzzles. Knights always tell the truth; knaves always lie. You will compare two attempts, A and B, that a small language model (the "solver") made on the same puzzle, and decide which attempt is better.

How to read an attempt
- The solver works in numbered steps "S1:", "S2:", ... (at most 5).
- In a step it sends requests; their results come back on the next line "R<k>:", in the same order, separated by " ; ":
  - "LLM <question>": a one-shot question to a helper model. The reply is short and may be wrong.
  - "SUB <question>": a sub-task solved by another copy of the solver; only its final result is shown, and it may be wrong.
  - Requests of the same step are separated by " ; ". "S<k>: -" means the step sent no requests.
- "ANSWER <text>" is the solver's final answer. It should give the role of every inhabitant.
- The requests and their results are the only information the solver had when it answered.

You are given the reference solution: the true role of every inhabitant. Use it to check every claim. Do not solve the puzzle again.

How to compare
1. Final answer. An attempt whose final answer is fully correct beats one whose answer is not.
2. If both final answers are wrong (or both are correct), weigh these two together:
   a. Partial correctness of the final answer: how many inhabitants it assigns correctly. A name that does not occur in the puzzle, a missing inhabitant, or an answer such as "contradiction" is wrong for that inhabitant; an answer that assigns no roles is no answer.
   b. Soundness of the process: requests that are relevant and well posed (for example, a case analysis on one inhabitant); results that are correct according to the reference; a final answer that actually follows from the results the attempt gathered; contradictions that are noticed and resolved rather than ignored.
   An attempt whose correct parts are supported by its own results is better than one that is right by luck.
3. Efficiency only breaks ties: fewer redundant or irrelevant requests. Do not prefer an attempt because it is longer or shorter, or because of its position (A or B).
If the attempts are equally good or equally useless, the verdict is TIE.

Reason briefly, in at most six short sentences: check each final answer against the reference, then the steps of each attempt. Then decide. The last two lines of your reply must be exactly:
VERDICT: <A, B or TIE>
CONFIDENCE: <an integer from 1 (unsure) to 5 (certain)>"""
JUDGE_PROMPT_ID = zlib.crc32(JUDGE_SYSTEM.encode())   # в кэше вердиктов: ответы на другую подсказку не считаются
VERDICT_RE = re.compile(r"VERDICT\s*:\s*\**\s*(A|B|TIE)\b", re.I)
CONF_RE = re.compile(r"CONFIDENCE\s*:\s*\**\s*([1-5])\b", re.I)


def judge_attempt_text(f):
    return "\n".join(step_line(i, st) for i, st in enumerate(f["steps"], 1)) or "(no steps)"


def judge_truth_text(truth):
    if isinstance(truth, dict):
        return ", ".join(f"{k}: {'knight' if v else 'knave'}" for k, v in truth.items())
    return str(truth)


def judge_messages(fa, fb):
    """Подсказка судье: условие, эталон, две попытки верхнего уровня (как их видит VM, без строки REQUEST)."""
    user = (f"PUZZLE:\n{fa['request']}\n\nREFERENCE SOLUTION:\n{judge_truth_text(fa['truth'])}\n\n"
            f"ATTEMPT A:\n{judge_attempt_text(fa)}\n\nATTEMPT B:\n{judge_attempt_text(fb)}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_verdict(text):
    """(A|B|TIE, уверенность 1–5) по последним строкам ответа; None — не разобрать."""
    v = VERDICT_RE.findall(text or "")
    if not v:
        return None
    c = CONF_RE.findall(text or "")
    return v[-1].upper(), (int(c[-1]) if c else 3)


def read_dotenv(name="OPENROUTER_API_KEY"):
    """Значение из .env (строка NAME=value, кавычки допустимы) рядом со скриптом или в текущей папке; .env — не в git."""
    for d in (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "", os.getcwd()):
        p = os.path.join(d, ".env")
        if d is not None and os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def openrouter_key():
    """Ключ OpenRouter: переменная окружения, затем .env (OPENROUTER_API_KEY=...), затем Colab Secrets (значок ключа
    слева, имя OPENROUTER_API_KEY). Нигде не печатается и не сохраняется."""
    k = os.environ.get("OPENROUTER_API_KEY", "").strip() or read_dotenv()
    if not k:
        try:
            from google.colab import userdata
            k = (userdata.get("OPENROUTER_API_KEY") or "").strip()
        except Exception:
            k = ""
    return k


def openrouter_chat(messages, model, key, max_tokens, extra=None, timeout=180, tries=4):
    """Один запрос к OpenRouter (urllib — без сторонних пакетов). (текст, usage). В ошибках — только код ответа."""
    import urllib.error
    import urllib.request
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                       "usage": {"include": True}, **(extra or {})}).encode("utf-8")
    err = "?"
    for k in range(tries):
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
            if out.get("error"):
                err = f"ошибка поставщика {str(out['error'].get('code', ''))[:20]}"
            else:
                msg = ((out.get("choices") or [{}])[0].get("message") or {})
                return (msg.get("content") or ""), (out.get("usage") or {})
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code in (400, 401, 402, 403, 404):
                break
        except Exception as e:
            err = type(e).__name__
        if k < tries - 1:
            time.sleep(min(30, 2 ** k) + random.random())
    raise RuntimeError(f"OpenRouter: {err}")


def stub_judge(messages):
    """Заглушка судьи (DRY_RUN): детерминированный вердикт по хешу подсказки, без сети."""
    h = zlib.crc32(messages[-1]["content"].encode())
    return f"stub\nVERDICT: {'ABT'[h % 3].replace('T', 'TIE')}\nCONFIDENCE: {1 + h % 5}", \
        {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.0001}


class JudgeRunner:
    """Запросы к судье в фоне (потоки; предл.): основной поток в это время собирает датасет на GPU. Готовые ответы
    забираются в основном потоке (collect) в кэш {ключ задания: ответ}; запись кэша на Drive — там же. Новые задания не
    отправляются, когда потрачено (с учётом ожидаемого на уже отправленные) ≥ judge_budget_usd."""
    def __init__(self, cfg, cache, path=None):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.cfg, self.cache, self.path = cfg, cache, path
        self.key = "" if cfg.dry_run else openrouter_key()
        self.pool = ThreadPoolExecutor(max_workers=max(1, cfg.judge_workers))
        self.pending, self.errors, self.capped = {}, Counter(), False
        self.lock, self.primary_fails, self.skip_until = threading.Lock(), 0, 0.0   # основная модель «отдыхает»
        self.done, self.last_save, self.seen = {}, time.time(), set()   # готовые ответы (из потоков) до collect
        self.closed = False

    def _finished(self, jk, fut):
        """Готовый ответ — в self.done и, не реже раза в минуту, весь кэш на Drive: обрыв сессии не теряет
        оплаченных вердиктов (запись из фонового потока, под замком; основной поток кэш только читает).
        Вызывается и из потока (callback), и из collect — обрабатывает ответ тот, кто первый."""
        with self.lock:
            if jk in self.seen or self.closed:
                return
            self.seen.add(jk)
        try:
            text, usage, sec, model = fut.result()
        except Exception as e:
            with self.lock:
                self.errors[str(e)[:60]] += 1
            return
        pv = parse_verdict(text)
        c = usage.get("cost")                            # нет цены в ответе — считаем по оценке, чтобы предел работал
        rec = {"v": pv[0] if pv else None, "c": pv[1] if pv else None, "model": model, "prompt": JUDGE_PROMPT_ID,
               "cost": float(c) if c else self.cfg.judge_cost_guess, "cost_est": not c,
               "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"), "sec": round(sec, 1),
               "text": text[-1500:]}
        with self.lock:
            if self.closed:
                return
            self.done[jk] = rec
            if self.path and time.time() - self.last_save > 60:
                self._flush()

    def _flush(self):
        """Под замком: готовые ответы — в кэш и кэш — на Drive."""
        self.cache.update(self.done)
        self.done.clear()
        self.last_save = time.time()
        if self.path:
            save_json(self.cache, self.path)

    def spent(self):
        with self.lock:
            return sum(v.get("cost", 0.0) for v in self.cache.values()) + sum(v["cost"] for v in self.done.values())

    def avg_cost(self):
        with self.lock:
            done = [v.get("cost", 0.0) for v in list(self.cache.values()) + list(self.done.values()) if "cost" in v]
        return (sum(done) / len(done)) if done else self.cfg.judge_cost_guess

    def submit(self, jobs):
        """jobs: [(ключ, messages)]; уже готовые и уже отправленные пропускаются. Ответы другой модели или другой
        подсказки судьи считаются отсутствующими (спрашиваются заново)."""
        with self.lock:
            stale = [jk for jk, m in jobs if jk in self.cache and
                     (self.cache[jk].get("prompt") != JUDGE_PROMPT_ID or self.cache[jk].get("model")
                      not in (self.cfg.judge_model, self.cfg.judge_fallback_model))]
            for jk in stale:
                del self.cache[jk]
        jobs = [(jk, m) for jk, m in jobs if jk not in self.cache and jk not in self.pending]
        if not jobs:
            return
        if not self.cfg.dry_run and not self.key:
            raise RuntimeError("нет ключа OpenRouter: положите его в .env рядом со скриптом (OPENROUTER_API_KEY=...), "
                               "в переменную окружения, в OPENROUTER_API_KEY в разделе 0 ноутбука или в Colab Secrets. "
                               "Без судьи: vm_pairs=\"R\" — пары только по R")
        spent, avg = self.spent(), self.avg_cost()
        for jk, msgs in jobs:
            if spent + (len(self.pending) + 1) * avg > self.cfg.judge_budget_usd:
                self.capped = True
                break
            with self.lock:
                self.seen.discard(jk)                    # повторная отправка после ошибки
            fut = self.pool.submit(self._one, msgs)
            self.pending[jk] = fut
            fut.add_done_callback(lambda f, jk=jk: self._finished(jk, f))

    def _one(self, msgs):
        """Основная модель; не ответила за judge_tries попыток — запасная (предл.). После 20 отказов основной подряд
        она 10 минут не спрашивается (лимит частоты у бесплатных моделей), запросы сразу идут к запасной."""
        t0, cfg = time.time(), self.cfg
        if cfg.dry_run:
            text, usage = stub_judge(msgs)
            return text, usage, time.time() - t0, cfg.judge_model
        fb = cfg.judge_fallback_model if cfg.judge_fallback_model != cfg.judge_model else ""
        if not fb or time.time() >= self.skip_until:
            try:
                text, usage = openrouter_chat(msgs, cfg.judge_model, self.key, cfg.judge_max_tokens,
                                              json.loads(cfg.judge_extra or "{}"), tries=cfg.judge_tries)
                with self.lock:
                    self.primary_fails = 0
                return text, usage, time.time() - t0, cfg.judge_model
            except RuntimeError:
                with self.lock:
                    self.primary_fails += 1
                    if self.primary_fails >= 20:
                        self.primary_fails, self.skip_until = 0, time.time() + 600
                if not fb:
                    raise
        text, usage = openrouter_chat(msgs, fb, self.key, cfg.judge_max_tokens,
                                      json.loads(cfg.judge_fallback_extra or "{}"), tries=4)
        return text, usage, time.time() - t0, fb

    def collect(self, wait=False, timeout=None):
        """Готовые ответы — в кэш и на Drive (wait=True — дождаться всех отправленных, но не дольше timeout с).
        Ошибка — задание не кэшируется (повторится при следующей отправке)."""
        from concurrent.futures import wait as fwait
        if wait and self.pending:
            fwait(list(self.pending.values()), timeout=timeout)
        for jk in [k for k, f in self.pending.items() if f.done()]:
            self._finished(jk, self.pending.pop(jk))     # если callback ещё не успел — обработать здесь
        with self.lock:
            self._flush()
        return len(self.pending)

    def close(self):
        """Кэш — на Drive; дальше ответы не принимаются и файл не трогается (новый JudgeRunner может писать тот же)."""
        with self.lock:
            self._flush()
            self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)


def judge_check(cfg):
    """Перед долгим прогоном: ключ есть, модель доступна, вердикт разбирается. Пара с очевидным победителем
    (верный ответ против «contradiction»), в обоих порядках; ≈$0.0002."""
    t = make_tasks(1, 11, cfg)[0]
    right = ", ".join(f"{k} is a {'knight' if v else 'knave'}" for k, v in t["truth"].items())
    fa = {"request": t["query"], "truth": t["truth"], "steps": [{"reqs": [], "answer": right}]}
    fb = {"request": t["query"], "truth": t["truth"], "steps": [{"reqs": [], "answer": "contradiction"}]}
    jr = JudgeRunner(cfg, {})
    jr.submit([("check:0", judge_messages(fa, fb)), ("check:1", judge_messages(fb, fa))])
    jr.collect(wait=True)
    jr.close()
    got = [jr.cache.get(k, {}).get("v") for k in ("check:0", "check:1")]
    res = {"judge_model": cfg.judge_model, "judge_check": got, "judge_check_cost": round(jr.spent(), 6),
           "judge_check_answered_by": [jr.cache.get(k, {}).get("model") for k in ("check:0", "check:1")]}
    log("проверка судьи:", res, "| ошибки:", dict(jr.errors) or "нет")
    if not cfg.dry_run:
        assert not jr.errors and None not in got, f"судья не ответил или ответ не разобрать: {dict(jr.errors)}"
        assert got == ["A", "B"], "судья не узнал верный ответ в очевидной паре — проверьте модель (judge_model)"
    return res


def dataset_pairs(frames):
    """Пары попыток одной задачи (верхний уровень): по task_id, если он есть (датасет, итерации), иначе по условию."""
    groups = {}
    for f in frames:
        if f["level"] == 0 and f["steps"]:
            groups.setdefault(f.get("task_id", f["request"]), []).append(f)
    out = []
    for g in groups.values():
        g = sorted(g, key=lambda f: f["id"])
        out += [(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))]
    return out


def pair_key(fa, fb):
    """Ключ пары: номер задачи и номера попыток (как в датасете); без них — id кадров."""
    if "task_id" in fa and "task_id" in fb:
        return f"task{fa['task_id']}:{fa.get('attempt')}-{fb.get('attempt')}"
    return f"{fa['id']}-{fb['id']}"


def job_key(fa, fb, o):
    """Ключ задания судье: пара, отпечаток текстов обеих попыток (новые попытки под тем же номером — другой ключ,
    старый вердикт к ним не пристанет) и порядок показа (0 — первая попытка как A, 1 — как B)."""
    return f"{pair_key(fa, fb)}:{zlib.crc32((vm_text(fa) + chr(0) + vm_text(fb)).encode()):08x}:{o}"


def judge_on(cfg):
    """Судья нужен, только если VM учится по его парам; подсказка судье — для K&K."""
    return cfg.vm_pairs == "judge" and cfg.task == "kk"


def judge_jobs(pairs, cfg):
    """Задания судье: пары с одинаковым исходом — в одном случайном порядке (на доле judge_both_frac — в обоих); пары
    с разным исходом — на доле judge_check_frac, как проверка согласия с R."""
    jobs = []
    for fa, fb in pairs:
        pk = pair_key(fa, fb)
        h = zlib.crc32(pk.encode())
        same = (fa.get("R", 0.0) > 0.5) == (fb.get("R", 0.0) > 0.5)
        if not same and (h >> 8) % 1000 >= cfg.judge_check_frac * 1000:
            continue
        orders = [h % 2] + ([1 - h % 2] if same and (h >> 4) % 1000 < cfg.judge_both_frac * 1000 else [])
        for o in orders:
            first, second = (fa, fb) if o == 0 else (fb, fa)
            jobs.append((job_key(fa, fb, o), judge_messages(first, second)))
    return jobs


def pair_labels(pairs, cache, cfg):
    """Пары для VM: [(текст победителя, текст проигравшего, вес, группа)] + сводка. Разный исход — победитель по R
    (вес 1). Одинаковый — по судье: TIE или разные вердикты в двух порядках — пара не берётся; вес — уверенность / 5."""
    out, st = [], Counter()
    for fa, fb in pairs:
        ra, rb = fa.get("R", 0.0) > 0.5, fb.get("R", 0.0) > 0.5
        verdicts = []
        for o in (0, 1):
            c = cache.get(job_key(fa, fb, o))
            if c and c.get("v"):
                first, second = (fa, fb) if o == 0 else (fb, fa)
                verdicts.append(({"A": first, "B": second}.get(c["v"]), c.get("c") or 3))
        gid = zlib.crc32(fa["request"].encode()) % 1000   # % 10 == 0 ⇔ held_task(request)
        if ra != rb:
            w, l = (fa, fb) if ra else (fb, fa)
            if cfg.vm_pairs in ("R", "judge"):
                out.append((vm_text(w), vm_text(l), 1.0, gid))
                st["pairs_by_R"] += 1
            for v, _ in verdicts:
                st["check: judge = R" if v is w else ("check: tie" if v is None else "check: judge != R")] += 1
            continue
        if not verdicts:
            st["no_verdict"] += 1
            continue
        if len(verdicts) == 2:
            st["both orders: " + ("agree" if verdicts[0][0] is verdicts[1][0] else "disagree")] += 1
        if any(v is None for v, _ in verdicts):
            st["tie"] += 1
            continue
        if len({id(v) for v, _ in verdicts}) > 1:
            st["orders_disagree_dropped"] += 1
            continue
        if cfg.vm_pairs == "judge":
            w = verdicts[0][0]
            l = fb if w is fa else fa
            out.append((vm_text(w), vm_text(l), float(np.mean([c for _, c in verdicts])) / 5, gid))
            st["pairs_by_judge"] += 1
    return out, dict(st)


# %% [markdown]
# ## 10. Обучение модели мира, RM и критика; актор в воображении (SCAR)
# * Воображение как в DreamerV3 (ваше, W5): старт — окна настоящих состояний, горизонт H; критик — по λ-возвратам на
#   воображаемом плюс немного на настоящем; актор — только на воображаемом.
# * Горизонт не длиннее оставшихся ходов уровня: на последнем ходе воображаемый эпизод кончается, как настоящий.
# * Критик: critic_steps шагов по мини-пачкам за раунд; актор: один проход мини-пачками (как в GRPO).
# * RSSM учится ещё и на точных контрфактах (предл.): запросы шага независимы, поэтому следующее состояние для неполной
#   коалиции строк известно без исполнения — это ровно те вопросы, которые ей задаёт игра шага.
# * SCAR (ваше: «actor учится на critic», игроки — действия шага без результатов): игроки — строки действия (запросы,
#   а также PLAN и ANSWER — предл.); v(S) = RM(l'_S) + γ·V(l'_S), l'_S — окно, которое RSSM без случайности
#   предсказывает, если отправить только строки из S (предл.). Актор (ваше): −Σ φ·log π + β·KL, каждый токен строки
#   получает φ этой строки; φ — в масштабе возвратов, как в DreamerV3 (предл.). Плечо full_adv (для сравнения, предл.):
#   A_t для всех токенов + λ·(φ − среднее φ шага).

# %%
def coalition_next(f, t, S, n_players, cfg):
    """Следующее состояние, если в шаге t отправить только строки-игроки из S. Без исполнения: запросы шага
    независимы и их результаты уже известны (предл.)."""
    st = f["steps"][t]
    has_plan = st.get("plan") is not None
    prev_plan = next((x["plan"] for x in reversed(f["steps"][:t]) if x.get("plan")), None)
    plan = st["plan"] if has_plan and 0 in S and st["plan"] else prev_plan
    if st.get("answer") is not None:
        step = {"reqs": [], "answer": st["answer"] if n_players - 1 in S else None}
    else:
        off = 1 if has_plan else 0
        step = {"reqs": [r for j, r in enumerate(st["reqs"]) if j + off in S], "answer": None}
    subs = st.get("subs_before")
    if subs is not None:                              # свой счётчик после шага: минус созданные в коалиции подзадачи
        subs -= sum(r["dest"] == "SUB" for r in step["reqs"])
    return render_state(f["request"], f["level"], plan, f["steps"][:t] + [step], cfg, f.get("path", ()), subs)


def transitions(frames, cfg=None, cf_per_step=0):
    """Настоящие переходы (s, a, s'). cf_per_step > 0 (предл.): ещё точные контрфакты из того же шага — пустое
    действие и до cf_per_step случайных неполных коалиций строк, т. е. ровно то, о чём RSSM спрашивают в игре шага."""
    out = [(st["state"], st["action"], st["next_state"], st.get("phi")) for f in frames for st in f["steps"]]
    if not cf_per_step:
        return out
    for f in frames:
        for t, st in enumerate(f["steps"]):
            players = parse_action(st["action"], f["level"], cfg)[3]
            k = len(players)
            if k == 0:
                continue
            proper = [S for r in range(1, k) for S in itertools.combinations(range(k), r)]
            for S in [()] + random.sample(proper, min(cf_per_step, len(proper))):
                out.append((st["state"], "\n".join(players[i] for i in S), coalition_next(f, t, S, k, cfg), None))
    return out


def wm_batch(codec, b, cfg):
    s_ids, s_pad = pad_batch([codec.state_ids(x[0], cfg) for x in b])
    a_ids, a_pad = pad_batch([codec.ids(x[1], cfg.max_action_tokens) for x in b])
    n_ids, n_pad = pad_batch([codec.state_ids(x[2], cfg) for x in b])
    ct = torch.tensor([codec.copy_target(x[0], cfg) for x in b], device=DEVICE)
    cn = torch.tensor([codec.copy_target(x[2], cfg) for x in b], device=DEVICE)
    phi = torch.tensor([x[3] if len(x) > 3 and x[3] is not None else float("nan") for x in b], device=DEVICE)
    return s_ids, s_pad, a_ids, a_pad, n_ids, n_pad, ct, cn, phi


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()


def train_wm(wm, opt, codec, trans, cfg, phase, steps=None, epochs=None):
    """phase copy: только энкодер учится копировать состояние в окно (2 прохода вместо 7 — втрое быстрее; декодер и
    RSSM учатся в основной фазе). phase main: потери модели мира; вес
    KL(энкодер ‖ RSSM) растёт от 0 за rep_warmup шагов (против схлопывания окна), копирование — с малым весом."""
    if not trans:
        return {}
    wm.train()
    if steps is None:
        steps = max(1, epochs * math.ceil(len(trans) / cfg.wm_batch))
    stats, pr, nan_run = [], Progress(f"модель мира ({phase})", steps), 0
    for i in range(steps):
        pr.tick(i)
        b = random.sample(trans, min(len(trans), cfg.wm_batch))
        if phase == "copy":
            w_copy, rep_w = 1.0, 0.0
        else:
            wm.main_steps += 1
            w_copy, rep_w = cfg.copy_anchor, cfg.rep_scale * min(1.0, wm.main_steps / max(1, cfg.rep_warmup))
        bt = wm_batch(codec, b, cfg)
        with autocast():
            loss, st = wm.loss(*bt[:8], phi=None if phase == "copy" else bt[8], w_copy=w_copy, rep_w=rep_w,
                               copy_only=phase == "copy")
        if not torch.isfinite(loss):                    # NaN/inf: шаг пропускается; подряд много — ошибка
            nan_run += 1
            if nan_run >= 5:
                raise RuntimeError("потеря модели мира — NaN/inf пять шагов подряд; пришлите лог")
            continue
        nan_run = 0
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        stats.append(st)
    wm.eval()
    return {k: round(float(np.mean([x[k] for x in stats])), 4) for k in stats[0]} if stats else {}


@torch.no_grad()
def wm_diag(wm, codec, trans, cfg, n=64):
    """Несёт ли окно состояние: CE декодера при шуме 0.9 с настоящим окном и с чужим (перемешанным) окном —
    win_info = разница (нат; около 0 — окно пустое); различие окон; доля непустых ячеек; совпадение RSSM без
    подсказок с энкодером на следующем шаге."""
    if len(trans) < 2:
        return {}
    wm.eval()
    b = random.sample(trans, min(n, len(trans)))
    s_ids, s_pad, a_ids, a_pad, n_ids, n_pad, ct, _, _ = wm_batch(codec, b, cfg)
    W = cfg.window
    wt = window_dist(wm.encode_logits(s_ids, s_pad).float(), 0.0).argmax(-1)
    content = ct != Vocab.EMPTY                         # позиции окна, где должно стоять содержимое состояния
    wn = window_dist(wm.encode_logits(n_ids, n_pad).float(), 0.0).argmax(-1)
    nopad = torch.zeros_like(wt, dtype=torch.bool)
    mask = (torch.rand(s_ids.shape, device=s_ids.device) < 0.9) & ~s_pad
    mask[:, 0] |= (mask.sum(1) == 0) & ~s_pad[:, 0]
    noised = s_ids.masked_fill(mask, Vocab.MASK)
    ce = {}
    for name, w in (("true", wt), ("shuffled", wt[torch.randperm(len(b), device=wt.device)])):
        logits = wm.dec([(w, SEG_WIN_IN, nopad), (noised, SEG_STATE, s_pad)], out=slice(W, None))
        ce[name] = F.cross_entropy(logits[mask].float(), s_ids[mask]).item()
    pred = wm.predict(wt, a_ids, a_pad, cfg.denoise_steps, greedy=True)
    changed = wt != wn                                    # позиции, где следующее окно отличается от прошлого
    return {"copy_acc": round(float((wt == ct)[content].float().mean()) if content.any() else 0.0, 3),
            "win_info": round(ce["shuffled"] - ce["true"], 3), "dec_ce_true": round(ce["true"], 3),
            "win_distinct": round(len({tuple(w) for w in wt.tolist()}) / len(b), 3),
            "win_filled": round(float((wt != Vocab.EMPTY).float().mean()), 3),
            "rssm_free_acc": round(float((pred == wn).float().mean()), 3),
            "rssm_copy_base": round(float((wt == wn).float().mean()), 3),          # столько даёт простое копирование
            "rssm_changed_acc": round(float((pred == wn)[changed].float().mean()) if changed.any() else 0.0, 3)}


def lambda_returns(rewards, conts, values, gamma, lam):
    """rewards[h], conts[h]: [B]; values[h]: [B] для h = 0..H (последнее — бутстрап)."""
    H = len(rewards)
    ret = [None] * H
    nxt = values[H]
    for h in reversed(range(H)):
        ret[h] = rewards[h] + gamma * conts[h] * ((1 - lam) * values[h + 1] + lam * nxt)
        nxt = ret[h]
    return ret


def token_weights(actor, gen, lines, base, extra):
    """Вес каждого токена действия: base для всех + extra[i] для токенов строки-игрока i."""
    w = [base] * len(gen)
    if not lines or not any(extra):
        return w
    pieces = actor.pieces(gen)
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    joined = "".join(pieces)
    spans = []
    for line in lines:
        k = joined.find(line)
        while k >= 0 and any(a <= k < b for a, b in spans if a >= 0):
            k = joined.find(line, k + 1)
        spans.append((k, k + len(line)) if k >= 0 else (-1, -1))
    for t, st in enumerate(starts):
        for i, (a, b) in enumerate(spans):
            if a <= st < b:
                w[t] += extra[i]
    return w


class RunStat:
    """Скользящий масштаб возвратов (P95 − P5, как в DreamerV3): им делятся и A_t, и φ."""
    def __init__(self, decay=0.99, ret_scale=None):
        self.decay, self.ret_scale = decay, ret_scale

    def upd_ret(self, rets):
        if rets.numel() < 2:
            return
        s = float(torch.quantile(rets, 0.95) - torch.quantile(rets, 0.05))
        self.ret_scale = s if self.ret_scale is None else self.decay * self.ret_scale + (1 - self.decay) * s


def actor_update(actor, opt, items, cfg):
    """items: (prompt, gen, веса токенов). Потеря: −Σ w·log π / N + β·KL(π ‖ π_ref) (k3 по выбранным токенам)."""
    if not items:
        return {}
    actor.train_mode(True)
    n_tok = sum(len(it[1]) for it in items)
    opt.zero_grad()
    kls, pg = [], 0.0
    for s in range(0, len(items), cfg.lp_batch):
        b = items[s:s + cfg.lp_batch]
        with torch.no_grad():
            ref = actor.seq_logprobs([x[0] for x in b], [x[1] for x in b], which="ref")
        cur = actor.seq_logprobs([x[0] for x in b], [x[1] for x in b])
        loss = 0.0
        for lp, rl, (_, _, w) in zip(cur, ref, b):
            wt = torch.tensor(w, device=lp.device, dtype=torch.float32)
            d = rl - lp
            kl = torch.exp(d) - d - 1
            loss = loss + (-(wt * lp).sum() + cfg.kl_beta * kl.sum()) / n_tok
            kls.append(kl.mean().item())
            pg += (wt * lp).sum().item()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.params(), 1.0)
    opt.step()
    actor.train_mode(False)
    return {"kl": float(np.mean(kls)), "pg": pg / n_tok}


@torch.no_grad()
def step_games(wm, critic, codec, games, cfg):
    """Игра шага для актора (SCAR; ваше: игроки — действия без результатов, оценка — критик).
    games: (окно l, строки-игроки, конец эпизода при S ∋ ANSWER?, ходы кончились?). Для каждой коалиции S:
    l'_S = RSSM(l, строки S) без случайности (разница значений — только от действия), v(S) = RM(l'_S) + γ·V(l'_S)
    (без V, если эпизод на этом кончается). Возвращает φ (точный Shapley) для каждого шага."""
    coal, offs = [], []
    for w, lines, ans_i, last in games:
        k = len(lines)
        subsets = [S for r in range(k + 1) for S in itertools.combinations(range(k), r)]
        offs.append((len(coal), subsets))
        coal += [(w, "\n".join(lines[i] for i in S), last or (ans_i is not None and ans_i in S)) for S in subsets]
    vals, cs = [], max(16, 32768 // cfg.window)        # пачка коалиций: при окне 512 — по 64 (память)
    for s0 in range(0, len(coal), cs):
        chunk = coal[s0:s0 + cs]
        a_ids, a_pad = pad_batch([codec.ids(c[1], cfg.max_action_tokens) for c in chunk])
        nw = wm.predict(torch.stack([c[0] for c in chunk]), a_ids, a_pad, cfg.denoise_steps, greedy=True)
        term = torch.tensor([float(c[2]) for c in chunk], device=DEVICE)
        vals += (wm.reward(nw) + cfg.gamma * (1 - term) * critic(nw).float()).tolist()
    return [shapley(len(g[1]), {S: vals[off + j] for j, S in enumerate(subsets)})
            for g, (off, subsets) in zip(games, offs)]


def actor_steps(actor, opt, items, cfg):
    """Один проход по своим (on-policy) примерам мини-пачками по actor_batch: столько же шагов на пример, что и в GRPO."""
    random.shuffle(items)
    stats = [actor_update(actor, opt, items[s:s + cfg.actor_batch], cfg) for s in range(0, len(items), cfg.actor_batch)]
    stats = [x for x in stats if x]
    out = {k: float(np.mean([x[k] for x in stats])) for k in stats[0]} if stats else {}
    out["actor_steps"] = len(stats)
    return out


def imagine_round(actor, actor_opt, wm, critic, critic_tgt, critic_opt, codec, starts, real_frames, rs, cfg):
    """Один раунд воображения: прогон до H шагов (с учётом оставшихся ходов), критик (λ-возвраты + немного
    настоящих), актор. starts: (текст состояния, сколько ходов уже сделано, уровень, остаток счётчика SUB)."""
    B = len(starts)
    w0 = encode_texts(wm, codec, [x[0] for x in starts], cfg)
    used, levels, subs = [x[1] for x in starts], [x[2] for x in starts], [x[3] for x in starts]
    ws, recs = [w0], []
    alive = torch.ones(B, dtype=torch.bool, device=DEVICE)
    for h in range(cfg.horizon):
        idx = alive.nonzero().flatten().tolist()
        if not idx:
            break
        cur = ws[-1]
        prompts = [actor_prompt(actor, cfg, codec.window_text(cur[b].tolist()), allow_sub=subs[b] > 0) for b in idx]
        acts = actor.generate(prompts, cfg.gen_action_tokens, greedy=False, temperature=cfg.temperature)
        a_ids, a_pad = pad_batch([codec.ids(t, cfg.max_action_tokens) for _, t in acts])
        nxt = cur.clone()
        nxt[idx] = wm.predict(cur[idx], a_ids, a_pad, cfg.denoise_steps)
        with torch.no_grad():
            r = torch.zeros(B, device=DEVICE)
            r[idx] = wm.reward(nxt[idx])
        c = torch.zeros(B, device=DEVICE)
        step = []
        for k, b in enumerate(idx):
            plan, reqs, ans, players = parse_action(acts[k][1], levels[b], cfg)
            if subs[b] <= 0:                               # счётчик кончился: SUB — обычным запросом
                for rq in reqs:
                    rq["dest"] = "LLM"
            subs[b] = max(0, subs[b] - sum(rq["dest"] == "SUB" for rq in reqs))
            last = used[b] + h + 1 >= cfg.max_steps            # ходы уровня кончились: эпизод обрывается
            step.append({"b": b, "prompt": prompts[k], "gen": acts[k][0], "players": players, "answer": ans,
                         "ans_i": len(players) - 1 if ans is not None else None, "last": last})
            c[b] = 0.0 if (ans is not None or last) else 1.0
            if c[b] == 0:
                alive[b] = False
        recs.append((step, r, c))
        ws.append(nxt)
    H = len(recs)
    if H == 0:
        return {}
    with torch.no_grad():
        vt = [critic_tgt(w).float() for w in ws]
        vc = [critic(w).float() for w in ws[:H]]
    rets = lambda_returns([x[1] for x in recs], [x[2] for x in recs], vt, cfg.gamma, cfg.lam_ret)
    # критик: λ-возвраты воображения
    pairs = [(h, s["b"]) for h, (step, _, _) in enumerate(recs) for s in step]
    win_b = torch.stack([ws[h][b] for h, b in pairs])
    tgt_b = torch.stack([rets[h][b] for h, b in pairs]).detach()
    # критик: немного настоящих траекторий (λ-возвраты по меткам RM)
    rw, tg_t = None, None
    real = [f for f in real_frames if f["steps"] and all("phi" in st for st in f["steps"])]
    if real and cfg.repval_scale > 0:
        real = random.sample(real, min(len(real), 64))
        texts = [st["state"] for f in real for st in f["steps"]]
        rw = encode_texts(wm, codec, texts, cfg)
        with torch.no_grad():
            rv = critic_tgt(rw).float()
        tg, k = [], 0
        for f in real:
            T = len(f["steps"])
            nxt_ret = 0.0
            out = [0.0] * T
            for t in reversed(range(T)):
                boot = rv[k + t + 1].item() if t + 1 < T else 0.0
                out[t] = f["steps"][t]["phi"] + cfg.gamma * ((1 - cfg.lam_ret) * boot + cfg.lam_ret * nxt_ret) \
                    if t + 1 < T else f["steps"][t]["phi"]
                nxt_ret = out[t]
            tg += out
            k += T
        tg_t = torch.tensor(tg, device=DEVICE)
    # критик: critic_steps шагов по мини-пачкам при неизменных целях раунда (генерация дороже, чем эти шаги)
    critic.train()
    losses_c = []
    for _ in range(cfg.critic_steps):
        i = torch.randint(0, len(pairs), (min(len(pairs), cfg.critic_batch),), device=DEVICE)
        loss_c = F.mse_loss(critic(win_b[i]).float(), tgt_b[i])
        if rw is not None:
            j = torch.randint(0, len(tg_t), (min(len(tg_t), cfg.critic_batch),), device=DEVICE)
            loss_c = loss_c + cfg.repval_scale * F.mse_loss(critic(rw[j]).float(), tg_t[j])
        critic_opt.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()
        losses_c.append(loss_c.item())
        with torch.no_grad():
            for p, q in zip(critic_tgt.parameters(), critic.parameters()):
                p.mul_(cfg.ema_decay).add_(q, alpha=1 - cfg.ema_decay)
    critic.eval()
    # актор. user_phi (ваше): −Σ φ·log π + β·KL, φ — Shapley строк шага по критику; adv_plus_centered (предл.,
    # для сравнения): A_t для всех токенов + scar_lambda·(φ_i − среднее φ) для строк. Всё в масштабе возвратов.
    rs.upd_ret(tgt_b)
    scale = max(1.0, rs.ret_scale or 1.0)
    flat = [(h, s) for h, (step, _, _) in enumerate(recs) for s in step]
    games = [(ws[h][s["b"]], s["players"], s["ans_i"], s["last"]) for h, s in flat if s["players"]]
    phis = iter(step_games(wm, critic, codec, games, cfg) if games else [])
    items, advs, phi_abs, sample = [], [], [], []
    for h, s in flat:
        phi = next(phis) if s["players"] else []
        a = float((rets[h][s["b"]] - vc[h][s["b"]]) / scale)
        if len(sample) < 16:                           # образцы для анализа: окно, строки действия, их φ и A_t
            sample.append({"h": h, "window": codec.window_text(ws[h][s["b"]].tolist())[-400:],
                           "players": s["players"], "phi": [round(x / scale, 4) for x in phi], "adv": round(a, 4),
                           "answer": s["answer"], "reward": round(float(recs[h][1][s["b"]]), 4)})
        if cfg.actor_credit == "user_phi":
            base, extra = 0.0, [x / scale for x in phi]
        else:
            mean = float(np.mean(phi)) if phi else 0.0
            base, extra = a, [cfg.scar_lambda * (x - mean) / scale for x in phi]
        items.append((s["prompt"], s["gen"], token_weights(actor, s["gen"], s["players"], base, extra)))
        advs.append(a)
        phi_abs += [abs(x) / scale for x in phi]
    st = actor_steps(actor, actor_opt, items, cfg)
    st.update({"critic_loss": float(np.mean(losses_c)), "imag_steps": len(pairs),
               "imag_answer_rate": float(np.mean([s["answer"] is not None for _, s in flat])),
               "adv_abs": float(np.mean(np.abs(advs))), "phi_abs": float(np.mean(phi_abs)) if phi_abs else 0.0,
               "multi_player_steps": sum(len(s["players"]) >= 2 for _, s in flat),
               "imag_reward": float(torch.stack([x[1] for x in recs]).mean()), "sample": sample})
    return st


# %% [markdown]
# ## 11. π0, базовая линия GRPO и оценка

# %%
def bc_items(actor, frames, cfg, mode, wm=None, codec=None):
    """Пары (подсказка, действие) из своих удачных попыток: наверху R = 1, подзадачи — внутри решённых задач."""
    steps = [st for f in frames if f.get("root_R", 0.0) > 0.5 for st in f["steps"]]
    if not steps:
        return []
    if mode == "window":
        wins = encode_texts(wm, codec, [st["state"] for st in steps], cfg)
        contents = [codec.window_text(w.tolist()) for w in wins]
    else:
        contents = [st["state"] for st in steps]
    def gen(st):                                       # действие с дописанным «= результат» — переразобрать начисто
        clean = clean_action(st["action"])
        return st["gen"] if clean == st["action"] else actor.encode_action(clean)
    return [(actor_prompt(actor, cfg, c, allow_sub=st.get("allow_sub", True)), gen(st))
            for c, st in zip(contents, steps)]


def bc_train(actor, items, cfg):
    if not items:
        log("  π0: удачных попыток нет — обучение по ним пропущено")
        return {}
    opt = torch.optim.AdamW(actor.params(), lr=cfg.bc_lr)
    actor.train_mode(True)
    losses, pr = [], Progress("π0", cfg.bc_epochs * math.ceil(len(items) / cfg.lp_batch))
    for _ in range(cfg.bc_epochs):
        random.shuffle(items)
        for s in range(0, len(items), cfg.lp_batch):
            pr.tick(len(losses))
            b = items[s:s + cfg.lp_batch]
            lps = actor.seq_logprobs([x[0] for x in b], [x[1] for x in b])
            loss = -torch.cat(lps).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.params(), 1.0)
            opt.step()
            losses.append(loss.item())
    actor.train_mode(False)
    return {"bc_loss": float(np.mean(losses)), "bc_items": len(items)}


def grpo_update(actor, opt, tops, frames, cfg):
    """GRPO без модели мира: все шаги всех уровней эпизода получают оценку эпизода (R против среднего по группе)."""
    groups = {}
    for f in tops:
        groups.setdefault(f["request"], []).append(f)
    adv = {}
    for g in groups.values():
        rs = np.array([f["R"] for f in g])
        for f in g:
            adv[f["id"]] = float((f["R"] - rs.mean()) / (rs.std() + 1e-6)) if rs.std() > 0 else 0.0
    byid = {f["id"]: f for f in frames}
    items = []
    for f in frames:
        root = f
        while root["parent"] is not None:
            root = byid[root["parent"][0]]
        a = adv.get(root["id"], 0.0)
        if a == 0.0:
            continue
        items += [(st["prompt"], st["gen"], [a] * len(st["gen"])) for st in f["steps"]]
    return {"grpo_items": len(items), **actor_steps(actor, opt, items, cfg)}


def evaluate(actor, tasks, cfg, mode, wm=None, codec=None, vm=None, few_shot=False, which="default", save=None):
    """Оценка на отложенных задачах (жадно); save — куда записать эпизоды (траектории) для анализа."""
    tops, frames = run_episodes(tasks, actor, cfg, mode, wm, codec, greedy=True, few_shot=few_shot, which=which)
    if save:
        save_json_gz(strip_frames(frames, ("prompt", "gen")), save)
    out = episode_stats(tops, frames)
    subs = [f for f in frames if f["level"] > 0 and f.get("sub_ok") is not None]
    out["sub_true_acc"] = float(np.mean([f["sub_ok"] for f in subs])) if subs else None   # только диагностика
    if vm is not None:
        out.update(vm_diagnostics(vm, frames))
    return out


def eval_direct(actor, tasks, save=None):
    """Базовая линия: та же модель решает задачу целиком с обычным рассуждением (ответ — после ANSWER:)."""
    prompts = [actor.prompt_ids(TASK.system_direct, t["query"]) for t in tasks]
    outs = actor.generate(prompts, TASK.direct_max_new, greedy=True, which="base")
    ok = [check_answer(o[1].split("ANSWER:")[-1], t["truth"]) for o, t in zip(outs, tasks)]
    if save:
        save_json_gz([{"query": t["query"], "truth": t["truth"], "ppl": t.get("ppl"), "output": o[1], "ok": bool(k)}
                      for t, o, k in zip(tasks, outs, ok)], save)
    out = {"acc": float(np.mean(ok))}
    by = {}
    for o, t in zip(ok, tasks):
        if t.get("ppl") is not None:
            by.setdefault(str(t["ppl"]), []).append(o)
    if by:
        out["by_ppl"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    return out


# %% [markdown]
# ## 12. Весь эксперимент
# Папка прогона на Drive (в ней README.txt — что где лежит):
# * **logs/** — всё лёгкое для анализа (текст и JSON; эту папку можно скачать целиком): log.txt, config.json,
#   results.json, evaluation/, trajectories/ (датасет, эпизоды итераций и оценок), judge/, <плечо>/metrics.json.
# * **weights/** — веса (нужны только для продолжения прогона): <плечо>/state.json, actor_initial/ (π0),
#   slot_1/ и slot_2/ — две копии весов, пишутся по очереди; state.json указывает на последнюю целую.

# %%
RUN_README = """Папка прогона sleepwalker. Две части: logs/ — всё лёгкое для анализа (её можно скачать целиком), weights/ —
веса моделей (тяжёлые; нужны только, чтобы продолжить прогон).

logs/
  README.txt              этот файл
  log.txt                 журнал прогона: всё, что печаталось
  pretraining_log.txt     (у эксперимента) журнал предобучения, с копии которого он стартовал; forked_from.json
  config.json             все настройки прогона
  results.json            итог: точности на отложенных задачах (по числу жителей тоже)
  task_source.json        откуда задачи: официальный набор Knights & Knaves или наш генератор
  vocabulary.json         словарь модели мира (нужен, чтобы читались её веса)
  adapter_check.json      итог быстрой проверки (ячейка 13)
  evaluation/
    tasks.json            отложенные задачи (из тестовой части набора; в обучении не участвуют)
    baselines.json        на них: прямой ответ модели и стартовая политика (без обучения)
  trajectories/           траектории — сжатый JSON (.json.gz). Кадр: запрос, уровень, путь сверху, шаги (state —
                          текст состояния, action — что написал актор, reqs — запросы и их результаты, answer),
                          ответ, R (верно ли наверху), root_R (решена ли задача наверху)
    dataset/              датасет: попытки базовой модели по задачам обучения (task_id — номер задачи, attempt —
                          номер попытки); part_000.json.gz … — части по мере сбора; settings.json — настройки сбора
    <плечо>/iteration_01.json.gz …   эпизоды итераций (каждая задача — iter_attempts попыток: task_id, attempt);
                          у шагов плеча с моделью мира phi — метка Shapley (от VM)
    <плечо>/dataset_with_shapley.json.gz   часть датасета, на которой училась модель мира, с метками Shapley
    <плечо>/imagination_01.json.gz …  образцы воображения (по 16 на раунд): окно, строки действия, их φ (Shapley по
                          критику) и преимущество A_t — то, на чём учится актор
    evaluation/           эпизоды оценок на отложенных задачах: base_model — стартовая политика; <плечо>_initial —
                          π0; <плечо>_iteration_05 — после 5-й итерации; direct_answer — прямые ответы модели.
                          У шагов плеча с моделью мира window — окно, которое видел актор
  judge/
    verdicts.json         ответы судьи по парам попыток датасета: ключ task<номер задачи>:<попытка>-<попытка>:<отпечаток
                          текстов обеих попыток, 8 hex>:<порядок> (:0 — первая попытка показана как A, :1 — как B):
                          вердикт v, уверенность c, модель, цена, рассуждение
    verdicts_<плечо>.json ответы судьи по парам попыток актора на итерациях (task_id = 1000000·итерация + номер задачи)
    summary.json          сводка по датасету: расход, согласие с R, сколько пар досталось VM
  <плечо>/metrics.json    метрики по итерациям

weights/
  <плечо>/state.json      сколько итераций сделано и какой слот — последняя целая копия
  <плечо>/actor_initial/  π0: актор после обучения на удачных попытках датасета (опора KL)
  <плечо>/slot_1/, slot_2/   две копии весов, пишутся по очереди (оборванная запись не портит последнюю целую):
                          actor/ — адаптер актора; value_model/ — VM; world_model.pt — модель мира (с RM) и критик

Плечи: world_model_seed0 — модель мира + Shapley (основной метод); grpo_seed0 — GRPO по полному тексту (сравнение).
"""

ARM_DIRS = {"full": "world_model", "full_adv": "world_model_adv", "grpo_text": "grpo"}


def arm_name(arm, seed):
    return f"{ARM_DIRS.get(arm, arm)}_seed{seed}"


class RunDirs:
    """Пути внутри папки прогона (см. RUN_README)."""
    def __init__(self, root):
        self.root = root
        self.logs = os.path.join(root, "logs")
        self.weights = os.path.join(root, "weights")
        self.traj = os.path.join(self.logs, "trajectories")
        self.dataset = os.path.join(self.traj, "dataset")
        self.judge = os.path.join(self.logs, "judge")
        self.evaluation = os.path.join(self.logs, "evaluation")

    def eval_traj(self, what):
        return os.path.join(self.traj, "evaluation", what + ".json.gz")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def window_gate(m, cfg, when, min_acc):
    """Ранняя остановка (предл.): всё дальше держится на окне. Окно пустое, если окна почти одинаковые или в них
    мало верно скопированного содержимого состояния (copy_acc — доля совпавших токенов, без пустых ячеек)."""
    bad = []
    if m.get("win_distinct", 1.0) < 0.5:
        bad.append(f"разных окон {m['win_distinct']:.2f} (нужно ≥ 0.5)")
    if m.get("copy_acc", 1.0) < min_acc:
        bad.append(f"скопировано верно {m['copy_acc']:.2f} содержимого (нужно ≥ {min_acc})")
    if bad and not cfg.dry_run:
        tail = ("долгий прогон не запускаю" if when == "на проверке" else
                "прогон остановлен, чтобы не тратить часы; датасет и оценки на Drive сохранены")
        raise RuntimeError(f"окно модели мира не обучилось ({when}): " + "; ".join(bad) + f". {tail}.")
    if bad:
        log("  (заглушки) окно не обучилось:", "; ".join(bad))


def next_slot(st):
    """Двойная запись (предл.): веса пишутся в слот, на который state.json не указывает, и только потом state.json
    переключается — оборванная запись не портит последнюю целую копию."""
    return "slot_2" if st.get("slot") == "slot_1" else "slot_1"


def window_examples(wm, codec, trans, cfg, n=2):
    wins = encode_texts(wm, codec, [t[0] for t in trans[:n]], cfg)
    return [(t[0][-160:], codec.window_text(w.tolist())[-160:]) for t, w in zip(trans[:n], wins)]


def attempt_trees(frames):
    """id верхнего кадра → кадры его попытки (вся рекурсия)."""
    byid = {f["id"]: f for f in frames}
    trees = {}
    for f in frames:
        g = f
        while g["parent"] is not None:
            g = byid[g["parent"][0]]
        trees.setdefault(g["id"], []).append(f)
    return trees


def reindex_frames(frames):
    """id → позиция в списке; ссылки родитель↔ребёнок переписываются (для части датасета)."""
    new = {f["id"]: i for i, f in enumerate(frames)}
    out = []
    for f in frames:
        g = dict(f, id=new[f["id"]])
        if g["parent"] is not None:
            g["parent"] = [new[g["parent"][0]]] + list(g["parent"][1:])
        g["steps"] = [dict(st, reqs=[dict(r, child=new[r["child"]]) if "child" in r else r for r in st["reqs"]])
                      for st in f["steps"]]
        out.append(g)
    return out


def wm_subset(data, cfg, seed=0):
    """Часть датасета для модели мира, RM и буфера (предл.): wm_attempts попыток целиком (0 — все). Шагов обучения
    модели мира столько же при любом объёме — лишние попытки дали бы только лишние часы игры VM и переходов."""
    trees = attempt_trees(data)
    roots = sorted(trees)
    if 0 < cfg.wm_attempts < len(roots):
        roots = sorted(random.Random(seed).sample(roots, cfg.wm_attempts))
    return reindex_frames([f for r in roots for f in sorted(trees[r], key=lambda f: f["id"])])


def success_frames(data, max_steps, seed=1):
    """Удачные попытки датасета целиком (для π0), не больше max_steps шагов."""
    trees = attempt_trees(data)
    byid = {f["id"]: f for f in data}
    ok = [r for r in sorted(trees) if byid[r].get("R", 0.0) > 0.5]
    random.Random(seed).shuffle(ok)
    out, n = [], 0
    for r in ok:
        k = sum(len(f["steps"]) for f in trees[r])
        if out and n + k > max_steps:
            break
        out += sorted(trees[r], key=lambda f: f["id"])
        n += k
    return reindex_frames(out)


def dataset_settings(cfg, n_attempts, n_parts):
    return json.loads(json.dumps({"attempts_total": n_attempts, "parts": n_parts, "cfg": {   # кортеж → список
        k: getattr(cfg, k) for k in ("task", "kk_source", "kk_people", "kk_depth", "dataset_tasks", "dataset_attempts",
                                     "dataset_parts", "dataset_temperature", "max_steps", "max_level", "max_reqs",
                                     "max_answer_chars", "gen_action_tokens", "actor_model", "sub_top", "sub_min",
                                     "sub_total")}}))


def normalize_frames(frames, cfg):
    """Кадры с Drive — в текущем формате состояния (предл.): тексты запросов и вопросы подзадач без дописанных
    «= …», пути пересобраны, состояния перерисованы. Так датасет, собранный прежним кодом (pro_v7, где в состояниях
    было «запрос = результат» и мусор в вопросах подзадач), читается тем же форматом, что и новые эпизоды."""
    byid = {f["id"]: f for f in frames}
    for f in frames:
        if f["level"] > 0:
            f["request"] = clean_request(f["request"])
        for st in f["steps"]:
            for r in st["reqs"]:
                r["text"] = clean_request(r["text"])
                if r["dest"] == "LLM" and "result" in r:
                    r["result"] = strip_echo(r["text"], r["result"])
    for f in frames:                                   # пути: родители идут раньше детей
        if f["parent"] is not None:
            f["path"] = context_block(byid[f["parent"][0]], f["parent"][1])
    for f in frames:
        plan = None
        for t, st in enumerate(f["steps"]):
            st["state"] = render_state(f["request"], f["level"], plan, f["steps"][:t], cfg, f["path"],
                                       st.get("subs_before"))
            plan = st.get("plan") or plan
        for t, st in enumerate(f["steps"]):
            st["next_state"] = (f["steps"][t + 1]["state"] if t + 1 < len(f["steps"]) else
                                render_state(f["request"], f["level"], f["plan"], f["steps"], cfg, f["path"],
                                             f["subs_left"]))
    return frames


def load_dataset(D, cfg):
    """Весь датасет из частей (id кадров — сквозные; части склеиваются всегда в одном порядке), в текущем формате."""
    st = load_json(os.path.join(D.dataset, "settings.json")) or {}
    return normalize_frames(merge_frames([load_json_gz(os.path.join(D.dataset, f"part_{k:03d}.json.gz"), [])
                                          for k in range(st.get("parts", 0))]), cfg)


def build_dataset(cfg, D, shard=None):
    """Датасет (ваше: все задачи обучения, по dataset_attempts попыток): базовая модель решает по полному тексту, с
    примерами; частями — каждая часть сразу на Drive (logs/trajectories/dataset/part_NNN.json.gz). Попытки одной задачи
    всегда в одной части. Судья (если включён) сравнивает пары готовых частей в фоне, пока собираются следующие.
    → (кадры, судья или None, актор или None)."""
    base = make_tasks(dataset_task_count(cfg), 1_000, cfg)
    tasks = [dict(t, task_id=i, attempt=a) for i, t in enumerate(base) for a in range(cfg.dataset_attempts)]
    per = math.ceil(len(base) / max(1, cfg.dataset_parts)) * cfg.dataset_attempts
    nparts = math.ceil(len(tasks) / per)
    settings = dataset_settings(cfg, len(tasks), nparts)
    sp = os.path.join(D.dataset, "settings.json")
    old = load_json(sp)
    if old is not None and {k: old.get(k) for k in settings} != settings:
        raise RuntimeError(f"датасет в {D.dataset} собран с другими настройками ({old.get('cfg')}): задайте новое имя "
                           f"прогона")
    if old is None:
        save_json(settings, sp)
    vp = os.path.join(D.judge, "verdicts.json")
    judge = JudgeRunner(cfg, load_json(vp, {}), vp) if judge_on(cfg) and shard is None else None   # шард: без судьи
    try:
        return build_dataset_(cfg, D, judge, base, tasks, per, nparts, settings, sp, old, shard)
    except BaseException:
        if judge is not None:
            judge.close()
        raise


def build_dataset_(cfg, D, judge, base, tasks, per, nparts, settings, sp, old, shard=None):
    """shard=(i, n) (предл.): собрать только части k ≡ i (mod n) — для нескольких GPU параллельно; судья и слияние —
    в основном запуске, когда все части готовы."""
    actor, parts = None, []
    for k in range(nparts):
        p = os.path.join(D.dataset, f"part_{k:03d}.json.gz")
        if shard is not None and k % shard[1] != shard[0]:
            continue
        part = load_json_gz(p)
        if part is None:
            check_stop(D)
            if actor is None:
                actor = make_actor(cfg)
                log(f"датасет: {len(base)} задач × {cfg.dataset_attempts} попытки, базовая модель по полному тексту "
                    f"(температура {cfg.dataset_temperature}); {nparts} частей по {per} попыток, каждая сразу "
                    f"пишется на Drive (готово частей: {k})" + ("; судья сравнивает пары в фоне" if judge else ""))
            _, frames = run_episodes(tasks[k * per:(k + 1) * per], actor, replace(cfg, temperature=cfg.dataset_temperature),
                                     "text", few_shot=True, which="base")
            part = strip_frames(frames)
            save_json_gz(part, p)
            log(f"  датасет: часть {k + 1}/{nparts} готова ({min(len(tasks), (k + 1) * per)}/{len(tasks)} попыток)"
                + (f"; судья: ответов {len(judge.cache)}, в работе {len(judge.pending)}, ${judge.spent():.3f}"
                   + (f", ошибок {sum(judge.errors.values())} {dict(judge.errors)}" if judge.errors else "")
                   if judge else ""))
        parts.append(part)
        if judge is not None:
            judge.collect()
            judge.submit(judge_jobs(dataset_pairs(part), cfg))
    if shard is not None:
        log(f"шард {shard[0]}/{shard[1]}: свои части готовы; остальные части и судья — в основном запуске")
        return None, None, actor
    data = normalize_frames(merge_frames(parts), cfg)
    if not (old or {}).get("complete"):
        settings.update(complete=True, stats=episode_stats([f for f in data if f["level"] == 0], data))
        save_json(settings, sp)
        log("  датасет готов:", settings["stats"])
    return data, judge, actor


def vm_trained(cfg, D):
    """Хоть у одного плеча уже пройдена стартовая цепочка (VM обучена): новые вердикты судьи ей не помогут."""
    return any((load_json(os.path.join(D.weights, arm_name(arm, seed), "state.json")) or {}).get("bc")
               for arm in cfg.arms for seed in cfg.seeds)


def finish_judge(cfg, D, data, judge):
    """Судья: отправить то, что ещё не отправлено (пока VM не обучена — потом новые вердикты ей не помогут),
    дождаться, сохранить ответы и сводку. → (пары для VM, сводка)."""
    vp = os.path.join(D.judge, "verdicts.json")
    allp = dataset_pairs(data)
    if judge is not None:
        if not vm_trained(cfg, D):
            judge.submit(judge_jobs(allp, cfg))
        if judge.pending:
            log(f"судья: жду {len(judge.pending)} ответов (не дольше 30 мин)")
        judge.collect(wait=True, timeout=1800)
        judge.close()
        cache = judge.cache
        fatal = {k: v for k, v in judge.errors.items() if any(x in k for x in ("HTTP 401", "HTTP 402", "HTTP 403"))}
        if fatal and not vm_trained(cfg, D):
            raise RuntimeError(f"судья: ключ или кредиты OpenRouter не в порядке ({fatal}). Датасет и готовые вердикты "
                               "на Drive; поправьте ключ/кредиты и запустите снова — недостающие пары доспросятся")
    else:
        cache = load_json(vp, {})
    pairs, st = pair_labels(allp, cache, cfg)
    vals = list(cache.values())
    chk = [st.get(k, 0) for k in ("check: judge = R", "check: judge != R", "check: tie")]
    summary = {"model": cfg.judge_model if judge is not None else None,
               "answers_by_model": dict(Counter(v.get("model", "?") for v in vals)), "requests": len(vals),
               "$": round(sum(v.get("cost", 0.0) for v in vals), 4),
               "tokens_per_request (in, out)": [round(float(np.mean([v.get("in") or 0 for v in vals])) if vals else 0),
                                                   round(float(np.mean([v.get("out") or 0 for v in vals])) if vals else 0)],
               "errors": dict(judge.errors) if judge is not None else {},
               "budget_cap_hit": bool(judge is not None and judge.capped),
               "agreement_with_R (mixed pairs)": round(chk[0] / sum(chk), 3) if sum(chk) else None,
               "pairs": st, "pairs_for_VM": len(pairs), "pairs_in_dataset": len(allp)}
    save_json(summary, os.path.join(D.judge, "summary.json"))
    log("судья и пары для VM:", summary)
    if sum(chk) >= 20 and chk[0] / sum(chk) < 0.9:
        log("!!! ВНИМАНИЕ: судья часто расходится с R там, где победитель известен — его пары для VM ненадёжны; "
            "смените judge_model или поставьте vm_pairs=\"R\"")
    same = sum(1 for fa, fb in allp if (fa.get("R", 0.0) > 0.5) == (fb.get("R", 0.0) > 0.5))
    if judge is not None and same and st.get("no_verdict", 0) > 0.05 * same:
        log(f"!!! ВНИМАНИЕ: у {st['без вердикта']} из {same} пар с одинаковым исходом нет вердикта судьи "
            f"(ошибки {dict(judge.errors)}, предел расхода {'достигнут' if judge.capped else 'нет'}): VM будет учиться "
            f"в основном на парах по R. Доспросить их можно, пока VM не обучена: остановите ячейку, проверьте "
            f"ключ/модель/предел (judge_budget_usd) и запустите снова. Если VM уже обучена — удалите "
            f"weights/<плечо>/state.json, тогда стартовая цепочка пройдёт заново")
    summary["complete"] = bool(judge is None or (not judge.pending and not judge.capped))
    save_json(summary, os.path.join(D.judge, "summary.json"))
    return pairs, summary


def run_full_(cfg, D, an, seed, boot_frames, bc_frames, pairs, vocab, eval_tasks, metrics, ref_acc, holder):
    dw, dl, dt = (os.path.join(x, an) for x in (D.weights, D.logs, D.traj))
    st = load_json(os.path.join(dw, "state.json"), {})
    vp = os.path.join(D.judge, f"verdicts_{an}.json")       # вердикты по парам попыток актора на итерациях (ваше)
    it_of = lambda k: int(k[4:].split(":")[0]) // 1_000_000 if k.startswith("task") else 0
    cache = {k: v for k, v in load_json(vp, {}).items() if it_of(k) <= st.get("it", 0)}   # оборванная итерация
    judge = JudgeRunner(cfg, cache, vp) if judge_on(cfg) else None   # идёт заново: её попытки будут другими
    holder["judge"] = judge
    metrics["iters"] = [m for m in metrics.get("iters", []) if m["it"] <= st.get("it", 0)]   # обрыв между записями
    save_metrics = lambda: save_json(metrics, os.path.join(dl, "metrics.json"))
    train_p, held_p = split_pairs(pairs)
    if not st.get("bc"):                               # стартовая цепочка: VM → метки RM → модель мира с RM → π0
        stage(f"{an}: предобучение VM")
        pre = os.path.join(dw, "pretraining")          # этапы цепочки сохраняются по ходу: обрыв не отбрасывает часы
        shapley_p = os.path.join(dt, "dataset_with_shapley.json.gz")
        saved = load_json_gz(shapley_p) if os.path.exists(os.path.join(pre, "value_model", "vm_meta.json")) and \
            "vm0" in metrics else None
        if saved and len(saved) == len(boot_frames) and all(a["request"] == b["request"] and len(a["steps"]) ==
                                                           len(b["steps"]) for a, b in zip(saved, boot_frames)):
            vm = make_vm(cfg, adapter=os.path.join(pre, "value_model"))
            for a, b in zip(saved, boot_frames):       # метки Shapley — из сохранённого
                for sa, sb in zip(a["steps"], b["steps"]):
                    if "phi" in sa:
                        sb["phi"] = sa["phi"]
            log(f"[{an}] VM и метки Shapley — с Drive (этап уже пройден):", metrics["vm0"])
        else:
            vm = make_vm(cfg)                          # VM первой (ваше): её Shapley — метки RM внутри модели мира
            metrics["vm0"] = {"pairs_acc_before": vm_pair_acc(vm, held_p)}
            point_ex = vm_examples([f for f in boot_frames if not held_task(f["root_query"])], cfg)   # без отложенных
            metrics["vm0"].update(train_vm_pairs(vm, train_p, point_ex, cfg, epochs=cfg.vm_pair_epochs))
            metrics["vm0"]["pairs_acc"] = vm_pair_acc(vm, held_p)
            metrics["vm0"].update(vm_game(vm, boot_frames, cfg))
            save_json_gz(strip_frames(boot_frames, ("prompt", "gen")), shapley_p)
            vm.save(os.path.join(pre, "value_model"))
            metrics.pop("wm_copy", None), metrics.pop("wm0", None)
            save_metrics()
            log(f"[{an}] VM (пар: {len(train_p)} для обучения, {len(held_p)} отложено; pairs_acc — доля отложенных "
                f"пар, где победитель оценён выше, 0.5 — наугад):", metrics["vm0"])
        stage(f"{an}: предобучение модели мира")
        actor = make_actor(cfg)
        codec = Codec(wm_tok(cfg), vocab)
        wm = WorldModel(vocab, cfg).to(DEVICE)
        wm.set_reward_scale(boot_frames)
        critic = ValueNet(vocab.size, cfg).to(DEVICE)
        wm_p = os.path.join(pre, "world_model.pt")
        if os.path.exists(wm_p) and "wm0" in metrics:
            ck = torch.load(wm_p, map_location=DEVICE)
            wm.load_state_dict(ck["wm"])
            wm.main_steps = ck.get("main_steps", 0)
            del ck
            wm.eval()
            log(f"[{an}] модель мира — с Drive (этап уже пройден):", metrics["wm0"])
        else:
            wm_opt = torch.optim.AdamW(wm.parameters(), lr=cfg.wm_lr)
            trans = transitions(boot_frames, cfg, cfg.cf_per_step)
            log(f"[{an}] модель мира: {len(trans)} переходов (с контрфактами) из {len(attempt_trees(boot_frames))} "
                f"попыток датасета; копирование {cfg.wm_copy_steps} шагов, затем {cfg.wm_boot_steps} шагов основной фазы")
            metrics["wm_copy"] = train_wm(wm, wm_opt, codec, trans, cfg, "copy", steps=cfg.wm_copy_steps)
            metrics["wm_copy"].update(wm_diag(wm, codec, trans, cfg))
            log("  окно после копирования:", metrics["wm_copy"])
            window_gate(metrics["wm_copy"], cfg, "после копирования", cfg.gate_copy_acc)
            metrics["wm0"] = train_wm(wm, wm_opt, codec, trans, cfg, "main", steps=cfg.wm_boot_steps)
            metrics["wm0"].update(wm_diag(wm, codec, trans, cfg))
            log("  модель мира:", metrics["wm0"])
            window_gate(metrics["wm0"], cfg, "после основной фазы", cfg.gate_copy_acc)
            for s_text, w_text in window_examples(wm, codec, random.sample(trans, min(2, len(trans))), cfg):
                log(f"  пример: состояние …{s_text!r}\n           окно …{w_text!r}")
            del trans, wm_opt
            save_torch({"wm": half(wm.state_dict()), "main_steps": wm.main_steps}, wm_p)
            save_metrics()
        stage(f"{an}: π0")
        metrics["bc"] = bc_train(actor, bc_items(actor, bc_frames, cfg, "window", wm, codec), cfg)
        log("  π0:", metrics["bc"])
        actor.save_adapter(os.path.join(dw, "actor_initial"))
        st = {"bc": True, "it": 0, "slot": next_slot({})}
        sd = os.path.join(dw, st["slot"])
        actor.save_adapter(os.path.join(sd, "actor"))
        vm.save(os.path.join(sd, "value_model"))
        save_torch({"wm": half(wm.state_dict()), "critic": critic.state_dict(), "critic_tgt": critic.state_dict(),
                    "main_steps": wm.main_steps}, os.path.join(sd, "world_model.pt"))
        save_metrics()
        save_json(st, os.path.join(dw, "state.json"))
        shutil.rmtree(pre, ignore_errors=True)         # этапы цепочки больше не нужны: веса — в слоте
        del actor, vm
        free_gpu()
    sd = os.path.join(dw, st["slot"])
    actor = make_actor(cfg, adapter=os.path.join(sd, "actor"), ref_adapter=os.path.join(dw, "actor_initial"))
    codec = Codec(wm_tok(cfg), vocab)
    vm = make_vm(cfg, adapter=os.path.join(sd, "value_model"))
    wm = WorldModel(vocab, cfg).to(DEVICE)
    critic, critic_tgt = (ValueNet(vocab.size, cfg).to(DEVICE) for _ in range(2))
    ck = torch.load(os.path.join(sd, "world_model.pt"), map_location=DEVICE)
    wm.load_state_dict(ck["wm"])
    wm.main_steps = ck.get("main_steps", 0)
    critic.load_state_dict(ck["critic"])
    critic_tgt.load_state_dict(ck["critic_tgt"])
    wm.eval(), critic.eval(), critic_tgt.eval()
    wm_opt = torch.optim.AdamW(wm.parameters(), lr=cfg.wm_lr)     # Adam модели мира не храним (~2 ГБ)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    if "critic_opt" in ck:
        critic_opt.load_state_dict(ck["critic_opt"])
    # состояние оптимизатора актора не сохраняется (≈0.6 ГБ на запись); после обрыва Adam начинает заново
    actor_opt = torch.optim.AdamW(actor.params(), lr=cfg.actor_lr)
    rs = RunStat(ret_scale=ck.get("ret_scale"))
    del ck
    if "eval_pi0" not in metrics:
        stage(f"{an}: оценка π0")
        metrics["eval_pi0"] = evaluate(actor, eval_tasks, cfg, "window", wm, codec, vm, save=D.eval_traj(f"{an}_initial"))
        log(f"[{an}] π0 (по окну):", metrics["eval_pi0"])
        if ref_acc and metrics["eval_pi0"]["acc"] < 0.5 * ref_acc:
            log(f"!!! ВНИМАНИЕ [{an}]: π0 по окну ({metrics['eval_pi0']['acc']:.3f}) намного хуже стартовой "
                f"политики по полному тексту ({ref_acc:.3f}). Окно теряет нужное (см. win_info выше: около 0 — "
                f"окно пустое). Обучение в воображении поверх такого π0 мало что покажет.")
        a = metrics["eval_pi0"].get("vm_auc_top")
        g = metrics["eval_pi0"].get("vm_gain_auc_top")
        if a is None or g is None:
            log(f"!!! ВНИМАНИЕ [{an}]: VM на отложенных задачах не проверить — среди них нет удач.")
        elif a < 0.55 or g < 0.55:
            log(f"!!! ВНИМАНИЕ [{an}]: VM на отложенных задачах почти не отличает удачи от неудач по шагам "
                f"(AUC по полному тексту {a:.2f}, по приросту от шагов {g:.2f}; 0.5 — наугад, ниже — наоборот). "
                f"Метки RM (Shapley по VM) будут шумом, и обучение актора в воображении мало что покажет.")
        save_metrics()
    replay_cache = {}
    traj = lambda k: os.path.join(dt, f"iteration_{k:02d}.json.gz")
    for it in range(st["it"], cfg.iterations):
        check_stop(D)
        stage(f"{an}: итерация {it + 1}/{cfg.iterations}")
        t0 = time.time()
        set_seed(seed * 1000 + it)
        base_tasks = make_tasks(cfg.n_iter_tasks, 10_000 + seed * 1000 + it, cfg)
        tasks = [dict(t, task_id=1_000_000 * (it + 1) + i, attempt=a)   # task_id — свой на итерацию: ключи пар не
                 for i, t in enumerate(base_tasks) for a in range(cfg.iter_attempts)]   # пересекаются с датасетом
        tops, frames = run_episodes(tasks, actor, cfg, "window", wm, codec, greedy=False)
        replay_cache[it + 1] = strip_frames(frames, ("prompt", "gen"))
        save_json_gz(replay_cache[it + 1], traj(it + 1))   # сразу: обрыв не теряет эпизоды
        online = [f for k in range(max(1, it + 2 - cfg.online_pair_iters), it + 2) for f in replay_cache.get(k, [])]
        if judge is not None:                           # судья на парах новых попыток (ваше) — в фоне, пока VM/WM учатся;
            judge.collect()                             # недостающие вердикты прошлых итераций доспрашиваются
            judge.submit(judge_jobs(dataset_pairs(online), cfg))
        replay = list(boot_frames)
        for k in range(max(1, it + 2 - cfg.replay_iters), it + 2):
            if k not in replay_cache:
                replay_cache[k] = normalize_frames(load_json_gz(traj(k), []), cfg)
            replay += replay_cache[k]
        for k in [k for k in replay_cache if k < it + 2 - cfg.replay_iters]:
            del replay_cache[k]
        m = {"it": it + 1, "rollout": episode_stats(tops, frames)}
        fresh = vm_examples(frames, cfg, levels=True)   # проверка VM на новых эпизодах — до обучения на них
        pre = vm_probs(vm, [e[0] for e in fresh]) if fresh else []
        held = [held_task(f["root_query"]) for f in frames if f["steps"]]   # задачи вне обучения VM — отдельно
        pre0 = vm_probs(vm, [vm_text(f, set()) for f in frames if f["steps"]]) if fresh else []
        m["vm_preq"] = {lv: [[round(p, 4), e[1], round(p0, 4), int(h)] for p, p0, e, h in zip(pre, pre0, fresh, held)
                             if e[3] == (lv == "top")] for lv in ("top", "sub")}
        if judge is not None:                           # дождаться вердиктов по новым парам (обычно уже готовы)
            judge.collect(wait=True, timeout=300)
        vc = judge.cache if judge is not None else {}
        new_pairs, pst = pair_labels(dataset_pairs(online), vc, cfg)
        new_pairs = split_pairs(new_pairs)[0]           # отложенные задачи — не в обучение (проверка pairs_acc честная)
        m["vm_online_pairs"] = {"pairs (buffer)": len(new_pairs), **pst}
        newp = dataset_pairs(replay_cache[it + 1])
        miss = pair_labels(newp, vc, cfg)[1].get("no_verdict", 0)
        m["vm_online_pairs"]["new_pairs_without_verdict"] = miss
        if judge is not None and miss > max(3, 0.3 * len(newp)):
            log(f"!!! ВНИМАНИЕ [{an}]: судья не ответил по {miss} из {len(newp)} новых пар (ошибки "
                f"{dict(judge.errors)}, предел {'достигнут' if judge.capped else 'нет'}) — они доспросятся на следующей "
                f"итерации; пока VM учится на остальных")
        m["vm"] = train_vm_pairs(vm, random.sample(train_p, min(len(train_p), cfg.vm_iter_pairs)) + new_pairs,
                                 vm_examples([f for f in replay if not held_task(f["root_query"])], cfg), cfg,
                                 budget=cfg.vm_iter_examples)
        m["vm"]["pairs_acc"] = vm_pair_acc(vm, held_p)
        m["vm"]["online_pairs"] = len(new_pairs)
        hist = [x.get("vm_preq", {}) for x in metrics.get("iters", [])] + [m["vm_preq"]]
        for lv in ("top", "sub"):                       # накопленный AUC по всем итерациям (0.5 — наугад)
            pq = [x for h in hist for x in h.get(lv, [])]
            for tag, sel in (("", pq), ("_heldout", [x for x in pq if len(x) > 3 and x[3]])):
                a = auc([x[0] for x in sel], [x[1] > 0.5 for x in sel])
                g = auc([x[0] - x[2] for x in sel], [x[1] > 0.5 for x in sel])
                m["vm"][f"new_auc_{lv}{tag}"] = None if a is None else round(a, 3)
                m["vm"][f"new_gain_auc_{lv}{tag}"] = None if g is None else round(g, 3)
                m["vm"][f"new_pos_{lv}{tag}"] = sum(x[1] > 0.5 for x in sel)
        m["vm"].update(vm_game(vm, replay, cfg, refresh=cfg.vm_refresh))
        save_json_gz(replay_cache[it + 1], traj(it + 1))   # ещё раз — уже с метками Shapley (phi) у шагов
        wm.set_reward_scale(replay)
        free_gpu()
        trans = transitions(replay, cfg, cfg.cf_per_step)
        m["wm"] = train_wm(wm, wm_opt, codec, trans, cfg, "main", steps=cfg.wm_iter_steps)
        m["wm"].update(wm_diag(wm, codec, trans, cfg))
        del trans
        free_gpu()
        starts = [(st_["state"], t, f["level"], st_.get("subs_before", 0)) for f in replay
                  for t, st_ in enumerate(f["steps"])]
        rounds = []
        for _ in range(cfg.imag_rounds):
            rounds.append(imagine_round(actor, actor_opt, wm, critic, critic_tgt, critic_opt, codec,
                                        random.sample(starts, min(len(starts), cfg.imag_batch)), replay, rs, cfg))
        samples = [r.pop("sample", []) for r in rounds]
        save_json_gz(samples, os.path.join(dt, f"imagination_{it + 1:02d}.json.gz"))   # образцы воображения
        m["imag"] = {k: float(np.mean([r[k] for r in rounds if k in r])) for k in (rounds[0] if rounds else {})}
        m["minutes"] = (time.time() - t0) / 60
        log(f"[{an}] итерация {it + 1}/{cfg.iterations}: успех попыток {m['rollout']['acc']:.3f} | "
            f"пары новых попыток {m['vm_online_pairs']} | VM {m['vm']} | WM с RM {m['wm']} | "
            f"воображение {m['imag']} | {m['minutes']:.1f} мин")
        if (it + 1) % cfg.eval_every == 0 or it + 1 >= cfg.iterations - 1:   # две последние — подряд (чередование)
            m["eval"] = evaluate(actor, eval_tasks, cfg, "window", wm, codec, vm,
                                 save=D.eval_traj(f"{an}_iteration_{it + 1:02d}"))
            log(f"[{an}] оценка после итерации {it + 1}:", m["eval"])
        metrics.setdefault("iters", []).append(m)
        new = {"bc": True, "it": it + 1, "slot": next_slot(st)}
        sd = os.path.join(dw, new["slot"])
        actor.save_adapter(os.path.join(sd, "actor"))
        vm.save(os.path.join(sd, "value_model"))
        save_torch({"wm": half(wm.state_dict()), "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                    "critic_opt": critic_opt.state_dict(),
                    "main_steps": wm.main_steps, "ret_scale": rs.ret_scale}, os.path.join(sd, "world_model.pt"))
        save_metrics()
        save_json(new, os.path.join(dw, "state.json"))
        st = new
    if judge is not None:
        judge.close()
        metrics["judge_online"] = {"requests": len(judge.cache), "$": round(judge.spent(), 4),
                                   "errors": dict(judge.errors), "budget_cap_hit": judge.capped}
        save_metrics()
    return metrics


def run_full(cfg, D, an, seed, boot_frames, bc_frames, pairs, vocab, eval_tasks, metrics, ref_acc=None):
    """run_full_ с гарантией: судья закрывается при любом выходе (иначе его потоки писали бы файл после ошибки)."""
    holder = {}
    try:
        return run_full_(cfg, D, an, seed, boot_frames, bc_frames, pairs, vocab, eval_tasks, metrics, ref_acc, holder)
    finally:
        if holder.get("judge") is not None:
            holder["judge"].close()


def run_grpo(cfg, D, an, seed, bc_frames, eval_tasks, metrics):
    dw, dl, dt = (os.path.join(x, an) for x in (D.weights, D.logs, D.traj))
    st = load_json(os.path.join(dw, "state.json"), {})
    metrics["iters"] = [m for m in metrics.get("iters", []) if m["it"] <= st.get("it", 0)]
    save_metrics = lambda: save_json(metrics, os.path.join(dl, "metrics.json"))
    if not st.get("bc"):
        actor = make_actor(cfg)
        metrics["bc"] = bc_train(actor, bc_items(actor, bc_frames, cfg, "text"), cfg)
        log(f"[{an}] π0 (по тексту):", metrics["bc"])
        actor.save_adapter(os.path.join(dw, "actor_initial"))
        st = {"bc": True, "it": 0, "slot": next_slot({})}
        actor.save_adapter(os.path.join(dw, st["slot"], "actor"))
        save_metrics()
        save_json(st, os.path.join(dw, "state.json"))
        del actor
        free_gpu()
    actor = make_actor(cfg, adapter=os.path.join(dw, st["slot"], "actor"),
                       ref_adapter=os.path.join(dw, "actor_initial"))
    opt = torch.optim.AdamW(actor.params(), lr=cfg.actor_lr)
    if "eval_pi0" not in metrics:
        metrics["eval_pi0"] = evaluate(actor, eval_tasks, cfg, "text", save=D.eval_traj(f"{an}_initial"))
        log(f"[{an}] π0 (по тексту):", metrics["eval_pi0"])
        save_metrics()
    for it in range(st["it"], cfg.iterations):
        check_stop(D)
        stage(f"{an}: итерация {it + 1}/{cfg.iterations}")
        t0 = time.time()
        set_seed(seed * 1000 + it)
        n_att = cfg.n_iter_tasks * cfg.iter_attempts          # столько же попыток, сколько у плеча с моделью мира
        base = make_tasks(max(1, n_att // cfg.grpo_group), 10_000 + seed * 1000 + it, cfg)
        tasks = [t for t in base for _ in range(cfg.grpo_group)]
        tops, frames = run_episodes(tasks, actor, cfg, "text", greedy=False)
        save_json_gz(strip_frames(frames, ("prompt", "gen")), os.path.join(dt, f"iteration_{it + 1:02d}.json.gz"))
        m = {"it": it + 1, "rollout": episode_stats(tops, frames), "grpo": grpo_update(actor, opt, tops, frames, cfg)}
        m["minutes"] = (time.time() - t0) / 60
        log(f"[{an}] итерация {it + 1}/{cfg.iterations}: успех попыток {m['rollout']['acc']:.3f} | "
            f"{m['grpo']} | {m['minutes']:.1f} мин")
        if (it + 1) % cfg.eval_every == 0 or it + 1 >= cfg.iterations - 1:
            m["eval"] = evaluate(actor, eval_tasks, cfg, "text", save=D.eval_traj(f"{an}_iteration_{it + 1:02d}"))
            log(f"[{an}] оценка после итерации {it + 1}:", m["eval"])
        metrics.setdefault("iters", []).append(m)
        new = {"bc": True, "it": it + 1, "slot": next_slot(st)}
        actor.save_adapter(os.path.join(dw, new["slot"], "actor"))
        save_metrics()
        save_json(new, os.path.join(dw, "state.json"))
        st = new
    return metrics


ARMS = {"full": {}, "full_adv": {"actor_credit": "adv_plus_centered"}}   # full_adv — сравнение (предл.)


@logged_errors
def run(cfg, name, shard=None):
    global CFG, LOG_PATH
    CFG = cfg
    set_task(cfg)
    D = RunDirs(os.path.join(BASE_DIR, name))
    os.makedirs(D.logs, exist_ok=True)
    os.makedirs(D.weights, exist_ok=True)
    LOG_PATH = os.path.join(D.logs, "log.txt")
    status_begin(D, name + (f" (шард {shard[0]}/{shard[1]})" if shard else ""))
    for rp in (os.path.join(D.root, "README.txt"), os.path.join(D.logs, "README.txt")):
        with open(rp, "w", encoding="utf-8") as f:
            f.write(RUN_README)
    save_json(asdict(cfg), os.path.join(D.logs, "config.json"))
    t0 = time.time()
    log(f"=== прогон {name} | {DEVICE} {DTYPE} | {json.dumps(asdict(cfg), ensure_ascii=False)}")
    src = task_source(cfg)
    meta = load_json(os.path.join(D.logs, "task_source.json"))
    if meta is None:
        save_json({"task": cfg.task, "source": src}, os.path.join(D.logs, "task_source.json"))
    elif meta.get("source") != src:
        raise RuntimeError(f"прогон {name} начинался на задачах «{meta.get('source')}», а сейчас доступны «{src}» "
                           "(официальный набор не загрузился?) — сравнение сломается; проверьте доступ к HF")
    eval_tasks = load_json(os.path.join(D.evaluation, "tasks.json"))
    if eval_tasks is None:
        set_seed(0)
        eval_tasks = make_tasks(cfg.n_eval, 2_000, cfg, split="test")
        save_json(eval_tasks, os.path.join(D.evaluation, "tasks.json"))
    stage("датасет" + (f", шард {shard[0]}/{shard[1]}" if shard else ""))
    data, judge, actor = build_dataset(cfg, D, shard)  # 0) датасет; судья — в фоне
    if shard is not None:
        STATUS["done"] = True
        write_status()
        return {}
    stage("базовые линии на отложенных задачах")
    bp = os.path.join(D.evaluation, "baselines.json")
    evals = load_json(bp, {})
    for key in ("direct", "text0"):                    # общие оценки; каждая сразу пишется на Drive
        if key not in evals:
            actor = actor or make_actor(cfg)
            evals[key] = eval_direct(actor, eval_tasks, save=D.eval_traj("direct_answer")) if key == "direct" else \
                evaluate(actor, eval_tasks, cfg, "text", few_shot=True, which="base", save=D.eval_traj("base_model"))
            log("  прямой ответ:" if key == "direct" else "  стартовая политика на отложенных:", evals[key])
            save_json(evals, bp)
            if judge is not None:                      # ответы судьи, пришедшие за это время, — в кэш и на Drive
                judge.collect()
    if actor is not None:
        del actor
        free_gpu()
    stage("судья: пары для VM")
    pairs, jsum = finish_judge(cfg, D, data, judge)
    boot = wm_subset(data, cfg)                        # часть датасета: модель мира, RM, буфер
    bc_frames = success_frames(data, cfg.bc_max_steps)  # удачные попытки: π0 обоих плеч
    n_att, n_ok = len(attempt_trees(data)), len(attempt_trees(bc_frames))
    del data
    vp = os.path.join(D.logs, "vocabulary.json")
    vocab_d = load_json(vp)
    if vocab_d is None:
        tok = wm_tok(cfg)
        texts = vocab_seed_texts(cfg) + [st["state"] for f in boot for st in f["steps"]] + \
            [st["action"] for f in boot for st in f["steps"]] + [st["next_state"] for f in boot for st in f["steps"]] + \
            [TASK.compact(st["state"]) for f in boot for st in f["steps"]]       # токены цели копирования
        vocab_d = {"itob": Vocab.build(texts, tok, cfg.vocab_cap).itob}
        save_json(vocab_d, vp)
    vocab = Vocab(vocab_d["itob"])
    log(f"словарь модели мира: {vocab.size} токенов; попыток в датасете {n_att}, из них для модели мира "
        f"{len(attempt_trees(boot))} ({len(boot)} кадров); удачных попыток для π0: {n_ok}")
    results = {"baselines": evals, "judge": jsum}
    for arm in cfg.arms:
        for seed in cfg.seeds:
            an = arm_name(arm, seed)
            stage(f"плечо {an}")
            metrics = load_json(os.path.join(D.logs, an, "metrics.json"), {})
            set_seed(seed)
            if arm in ARMS:
                ref = evals.get("text0", {}).get("acc")
                metrics = run_full(replace(cfg, **ARMS[arm]), D, an, seed, boot, bc_frames, pairs, vocab, eval_tasks,
                                   metrics, ref)
            elif arm == "grpo_text":
                metrics = run_grpo(cfg, D, an, seed, bc_frames, eval_tasks, metrics)
            results[an] = metrics
            free_gpu()
    nan = float("nan")
    lines = [f"прямой ответ: {evals.get('direct', {}).get('acc', nan):.3f}",
             f"стартовая политика (полный текст, без обучения): {evals.get('text0', {}).get('acc', nan):.3f}"]
    short = {"baselines": {k: {"acc": v.get("acc"), "by_ppl": v.get("by_ppl")} for k, v in evals.items()},
             "judge": {k: jsum.get(k) for k in ("model", "$", "pairs_for_VM", "agreement_with_R (mixed pairs)")}}
    for an, m in results.items():
        if an in ("baselines", "judge"):
            continue
        evs = [(x["it"], x["eval"]) for x in m.get("iters", []) if "eval" in x]
        lines.append(f"{an}: π0 {m.get('eval_pi0', {}).get('acc', nan):.3f} → "
                     + ", ".join(f"ит.{i}: {e['acc']:.3f}" for i, e in evs))
        short[an] = {"initial": {k: m.get("eval_pi0", {}).get(k) for k in ("acc", "by_ppl")},
                     **{f"iteration_{i:02d}": {k: e.get(k) for k in ("acc", "by_ppl")} for i, e in evs}}
    results["minutes_total"] = short["minutes_total"] = (time.time() - t0) / 60
    short["forked_from"] = (load_json(os.path.join(D.logs, "forked_from.json")) or {}).get("forked_from")
    save_json({**(load_json(os.path.join(D.logs, "results.json")) or {}), **short}, os.path.join(D.logs, "results.json"))
    log("ИТОГ (точность на отложенных задачах):\n  " + "\n  ".join(lines))   # плечи на разных GPU дописывают свой итог
    log(f"готово за {results['minutes_total']:.1f} мин; всё в {D.root} (лёгкое для анализа — в logs/)")
    STATUS["done"] = True
    write_status()
    return results


def free_gpu():
    """Вернуть память GPU: сначала сборщик мусора (у моделей HF циклические ссылки — без него удалённая модель
    может ещё долго занимать память), потом кэш CUDA."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


@logged_errors
def adapter_check(cfg, name):
    """Быстрая проверка без долгого прогона (≈1–3 мин; один раз на прогон). Актор делает несколько шагов обучения,
    адаптер пишется на Drive и читается обратно: log π должны совпасть. Тот же адаптер, подключённый как ref (π0),
    тоже должен совпасть, а без адаптера должна получиться исходная модель. То же для VM с головой."""
    global CFG, LOG_PATH
    CFG = cfg
    set_task(cfg)
    D = RunDirs(os.path.join(BASE_DIR, name))
    os.makedirs(D.logs, exist_ok=True)
    LOG_PATH = os.path.join(D.logs, "log.txt")
    status_begin(D, name)
    stage("быстрая проверка")
    mark = os.path.join(D.logs, "adapter_check.json")
    if (load_json(mark) or {}).get("version") == 7:  # версия проверки: новая проверка не пропускается
        log("проверка адаптеров для этого прогона уже пройдена:", load_json(mark))
        return load_json(mark)
    tmp = os.path.join(D.weights, "adapter_check_tmp")
    res = judge_check(cfg) if judge_on(cfg) else {}    # первым: без ключа — ошибка сразу, а не через минуты
    shots = task_shots(cfg)
    # актор: несколько шагов обучения → запись → чтение
    a1 = make_actor(cfg)
    probe = [actor_prompt(a1, cfg, shots[0][0])]
    target = [a1.text_tok.encode(shots[0][1]) + [a1.tok.eos_token_id]]
    with torch.no_grad():
        lp_base = a1.seq_logprobs(probe, target, which="base")[0]
    opt = torch.optim.AdamW(a1.params(), lr=1e-3)
    a1.train_mode(True)
    for _ in range(5):
        loss = -a1.seq_logprobs(probe, target)[0].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    a1.train_mode(False)
    with torch.no_grad():
        lp1 = a1.seq_logprobs(probe, target)[0]
    a1.save_adapter(os.path.join(tmp, "actor"))
    del a1, opt
    free_gpu()
    a2 = make_actor(cfg, adapter=os.path.join(tmp, "actor"), ref_adapter=os.path.join(tmp, "actor"))
    with torch.no_grad():
        lp2, lp_ref, lp_b2 = [a2.seq_logprobs(probe, target, which=w)[0] for w in ("default", "ref", "base")]
    # generate на настоящей модели (до долгого прогона): разные длины подсказок, жадно, с выборкой, без адаптера.
    # Первая и последняя подсказки одинаковы: при жадной генерации ответы должны совпасть (порядок восстановлен)
    prompts = [actor_prompt(a2, cfg, u) for u, _ in shots] + probe
    for greedy, which in ((True, "default"), (False, "default"), (True, "base")):
        outs = a2.generate(prompts, 24, greedy=greedy, which=which)
        assert len(outs) == len(prompts) and all(len(g) > 0 and isinstance(t, str) for g, t in outs), "generate сломан"
        if greedy and not cfg.dry_run:
            assert outs[0][0] == outs[-1][0], "generate перепутал порядок ответов"
    log("  generate: OK (жадно, с выборкой, без адаптера; порядок ответов сохранён)")
    del a2
    free_gpu()
    diff = lambda x, y: (x.float() - y.float()).abs().max().item()
    res.update({"actor_moved": diff(lp1, lp_base), "actor_reload": diff(lp2, lp1), "actor_ref": diff(lp_ref, lp1),
                "actor_base": diff(lp_b2, lp_base)})
    tol = min(0.05, res["actor_moved"] / 4)          # грубая ошибка (адаптер не подключён) даёт расхождение = moved
    log("проверка адаптера актора:", {k: f"{res[k]:.2e}" for k in ("actor_moved", "actor_reload", "actor_ref", "actor_base")},
        f"| допуск {tol:.2e}")
    assert res["actor_moved"] > 1e-3, "несколько шагов обучения не изменили актора — адаптер не учится"
    assert res["actor_reload"] < tol, "адаптер актора, прочитанный с Drive, даёт другие log π"
    assert res["actor_ref"] < tol, "адаптер, подключённый как ref (π0), даёт другие log π"
    assert res["actor_base"] < tol, "без адаптера получается не исходная модель"
    # VM: шаг обучения → запись → чтение
    texts = [u for u, _ in shots[:2]]
    v1 = make_vm(cfg)
    v1.eval()
    with torch.no_grad():
        r0 = v1(texts).float()                          # сырые логиты: двигает ли VM обучение
    train_vm(v1, [(texts[0], 1.0, 1.0), (texts[1], 0.0, 1.0), (texts[1], 0.0, 1.0)], cfg)
    off = list(v1.logit_offset)
    assert len(off) == 2 and all(math.isfinite(x) for x in off), "сдвиги логита VM не подобраны"
    with torch.no_grad():
        r1 = v1(texts).float()
        o1 = r1 - torch.tensor([vm_offset_of(v1, t) for t in texts], device=r1.device)   # то, что идёт в P успеха
    v1.save(os.path.join(tmp, "vm"))
    del v1
    free_gpu()
    v2 = make_vm(cfg, adapter=os.path.join(tmp, "vm"))
    v2.eval()
    assert all(abs(a - b) < 1e-9 for a, b in zip(v2.logit_offset, off)), "сдвиги логита VM не пережили запись/чтение"
    with torch.no_grad():
        r2 = v2(texts).float()
        o2 = r2 - torch.tensor([vm_offset_of(v2, t) for t in texts], device=r2.device)
    del v2
    free_gpu()
    res.update({"vm_moved": diff(r1, r0), "vm_reload": diff(o2, o1)})
    tol_vm = min(0.05, res["vm_moved"] / 4)
    log("проверка VM:", {k: f"{res[k]:.2e}" for k in ("vm_moved", "vm_reload")}, f"| допуск {tol_vm:.2e}")
    assert res["vm_moved"] > 1e-3, "шаг обучения не изменил VM"
    assert res["vm_reload"] < tol_vm, "VM, прочитанная с Drive, даёт другие оценки"
    res.update(wm_check(cfg))
    res.update(ok=True, version=7)
    save_json(res, mark)
    shutil.rmtree(tmp, ignore_errors=True)
    log("ПРОВЕРКА АДАПТЕРОВ: OK")
    return res


@torch.no_grad()
def copy_eval(wm, codec, trans, cfg, n=64):
    """Потеря копирования на случайной пачке, без обучения."""
    wm.eval()
    with autocast():
        _, st = wm.loss(*wm_batch(codec, random.sample(trans, min(n, len(trans))), cfg), w_copy=1.0, rep_w=0.0)
    return st["copy"]


def wm_check(cfg, steps=300):
    """Модель мира до долгого прогона (≈2–3 мин): на правдоподобных состояниях задачи окно должно научиться
    копировать состояние. Иначе актор прочтёт в окне бессмыслицу — прогон останавливается."""
    rng = random.Random(5)
    trans = []
    for t in make_tasks(64, 7, cfg):                  # состояния: запрос (иногда — подзадача) + до 3 шагов
        q, level, path = t["query"], 0, []
        if rng.random() < 0.3:                          # подзадача: путь — исходная задача уровнем выше
            q, level, path = "Check the first part of the task.", 1, [f"[L0] Q: {t['query']}"]
        steps_, plan, subs = [], None, sub_allow(cfg, level)
        for k in range(rng.randint(1, 4)):
            state = render_state(q, level, plan, steps_, cfg, path, subs)
            text = TASK.stub_action(state, rng, cfg)
            plan_k, reqs, ans, _ = parse_action(text, level, cfg)
            for r in reqs:
                if r["dest"] == "SUB" and subs <= 0:
                    r["dest"] = "LLM"
                subs -= r["dest"] == "SUB"
                r["result"] = TASK.norm_result(TASK.stub_exec(r["text"], rng))
            steps_.append({"reqs": reqs, "answer": ans})
            plan = plan_k or plan
            trans.append((state, text, render_state(q, level, plan, steps_, cfg, path, subs)))
            if ans is not None:
                break
    tok = wm_tok(cfg)
    vocab = Vocab.build(vocab_seed_texts(cfg) + [x for tr in trans for x in tr], tok, cfg.vocab_cap)
    codec = Codec(tok, vocab)
    wm = WorldModel(vocab, cfg).to(DEVICE)
    opt = torch.optim.AdamW(wm.parameters(), lr=cfg.wm_lr)
    c0 = copy_eval(wm, codec, trans, cfg)
    train_wm(wm, opt, codec, trans, cfg, "copy", steps=steps)
    m = {"copy": copy_eval(wm, codec, trans, cfg), **wm_diag(wm, codec, trans, cfg)}
    log(f"проверка модели мира ({cfg.wm_backbone if not cfg.dry_run else 'заглушка'}): copy {c0:.2f} → "
        f"{m['copy']:.2f} за {steps} шагов | скопировано верно {m['copy_acc']:.2f} |",
        {k: m[k] for k in ("win_distinct", "win_filled", "win_info")})
    for s_text, w_text in window_examples(wm, codec, random.Random(1).sample(trans, 2), cfg):
        log(f"  пример: состояние …{s_text!r}\n           окно …{w_text!r}")
    window_gate(m, cfg, "на проверке", cfg.gate_copy_acc_check)
    del wm, opt
    free_gpu()
    return {"wm_copy_start": c0, "wm_copy_end": m["copy"], "wm_copy_acc": m["copy_acc"],
            "wm_win_distinct": m["win_distinct"]}


@logged_errors
def fork_run(src, dst):
    """Эксперимент стартует с копии предобучения (датасет, вердикты судьи, оценки, словарь, π0 и модель мира обоих плеч):
    копируется один раз; дальше папка эксперимента живёт сама. Одно предобучение — сколько угодно экспериментов,
    если в них меняется только обучение с подкреплением (итерации, шаги актора и т. п.)."""
    s, d = os.path.join(BASE_DIR, src), os.path.join(BASE_DIR, dst)
    if os.path.exists(d):
        return
    S = RunDirs(s)
    ready = (load_json(os.path.join(S.dataset, "settings.json")) or {}).get("complete") and os.path.isdir(S.weights) \
        and any((load_json(os.path.join(S.weights, a, "state.json")) or {}).get("bc") for a in os.listdir(S.weights))
    assert ready, f"предобучение {src} не готово"
    shutil.rmtree(d + ".tmp", ignore_errors=True)       # недокопированное после обрыва — заново
    shutil.copytree(s, d + ".tmp", ignore=shutil.ignore_patterns("log.txt", "results.json", "adapter_check_tmp"))
    if os.path.exists(os.path.join(S.logs, "log.txt")):   # журнал предобучения — рядом, чтобы logs/ был полным
        shutil.copy2(os.path.join(S.logs, "log.txt"), os.path.join(d + ".tmp", "logs", "pretraining_log.txt"))
    save_json({"forked_from": src}, os.path.join(d + ".tmp", "logs", "forked_from.json"))
    os.rename(d + ".tmp", d)                            # папка появляется только целиком
    line = f"эксперимент {dst} начат с копии предобучения {src}"
    log(line)
    with open(os.path.join(d, "logs", "log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("[%H:%M:%S] ") + line + "\n")


def check_layout(root):
    """Раскладка папки прогона: всё нужное на месте; в logs/ — только лёгкое (текст, JSON, .json.gz), в weights/ —
    без траекторий; в траекториях итераций плеча с моделью мира — метки Shapley."""
    need = ["README.txt", "logs/README.txt", "logs/log.txt", "logs/status.json", "logs/config.json", "logs/results.json",
            "logs/task_source.json", "logs/vocabulary.json", "logs/evaluation/tasks.json",
            "logs/evaluation/baselines.json", "logs/judge/summary.json",
            "logs/trajectories/dataset/settings.json", "logs/trajectories/dataset/part_000.json.gz",
            "logs/trajectories/evaluation/base_model.json.gz", "logs/trajectories/evaluation/direct_answer.json.gz",
            "logs/trajectories/world_model_seed0/imagination_01.json.gz",
            "logs/trajectories/evaluation/world_model_seed0_initial.json.gz",
            "logs/trajectories/evaluation/grpo_seed0_initial.json.gz",
            "logs/trajectories/world_model_seed0/iteration_01.json.gz",
            "logs/trajectories/world_model_seed0/dataset_with_shapley.json.gz",
            "logs/trajectories/grpo_seed0/iteration_01.json.gz", "logs/world_model_seed0/metrics.json",
            "logs/grpo_seed0/metrics.json", "weights/world_model_seed0/state.json",
            "weights/world_model_seed0/actor_initial", "weights/grpo_seed0/state.json", "weights/grpo_seed0/actor_initial"]
    missing = [x for x in need if not os.path.exists(os.path.join(root, x))]
    assert not missing, f"{root}: нет {missing}"
    D = RunDirs(root)
    heavy = [os.path.join(dp, f) for dp, _, fs in os.walk(D.logs) for f in fs if not f.endswith((".json", ".txt", ".json.gz"))]
    assert not heavy, f"в logs/ не только лёгкое: {heavy[:5]}"
    traj = [os.path.join(dp, f) for dp, _, fs in os.walk(D.weights) for f in fs if f.endswith(".json.gz")]
    assert not traj, f"траектории в weights/: {traj[:5]}"
    it1 = load_json_gz(os.path.join(root, "logs/trajectories/world_model_seed0/iteration_01.json.gz"))
    assert any("phi" in st for f in it1 for st in f["steps"]), "в траекториях итерации нет меток Shapley"
    assert all("window" in st for f in it1 for st in f["steps"]), "в траекториях итерации нет окна актора"
    assert not any("window" in st for f in load_json_gz(os.path.join(root, "logs/trajectories/grpo_seed0/iteration_01.json.gz"))
                   for st in f["steps"]), "у GRPO (по тексту) окна быть не должно"
    if os.path.exists(os.path.join(root, "logs/forked_from.json")):
        assert os.path.exists(os.path.join(root, "logs/pretraining_log.txt")), "у эксперимента нет журнала предобучения"
        assert not os.path.isdir(os.path.join(root, "weights/world_model_seed0/pretraining")), "этапы предобучения не удалены"
    for arm in ("world_model_seed0", "grpo_seed0"):
        st = load_json(os.path.join(D.weights, arm, "state.json"))
        assert os.path.isdir(os.path.join(D.weights, arm, st["slot"], "actor")), (arm, st)
    st = load_json(os.path.join(D.weights, "world_model_seed0", "state.json"))
    for x in ("value_model", "world_model.pt"):
        assert os.path.exists(os.path.join(D.weights, "world_model_seed0", st["slot"], x)), x


def dry_check():
    """Только локально (DRY_RUN=1 python3 sleepwalker.py): весь конвейер на заглушках, без GPU, сети и моделей.
    Проверяет: запись/чтение адаптеров и VM; метки VM; судью (заглушка) и пары; попарное обучение VM; сбор датасета
    частями и продолжение после обрыва; продолжение прогона с Drive; игру шага (SCAR); GRPO; контрфакты; предобучение
    и эксперимент с его копии; раскладку папки прогона (logs/ — только лёгкое, weights/ — веса)."""
    assert adapter_check(Config.dry(), "dry_test")["ok"] and adapter_check(Config.dry(), "dry_test")["ok"]
    ans, none = {"reqs": [], "answer": "x"}, {"reqs": [], "answer": None}   # метки VM: ответ подзадачи ≠ «полезно»
    fr = [{"level": 0, "steps": [ans], "R": 0.0, "request": "q", "answer": "x"},
          {"level": 1, "steps": [ans], "root_R": 0.0, "request": "q", "answer": "x", "path": ["p"]},
          {"level": 1, "steps": [ans], "root_R": 1.0, "request": "q", "answer": "x", "path": ["p"]},
          {"level": 1, "steps": [none], "root_R": 1.0, "request": "q", "answer": None, "path": ["p"]},
          {"level": 1, "steps": [], "root_R": 1.0, "request": "q", "answer": None, "path": ["p"]}]
    sw = CFG.sub_weight
    assert [(y, w) for _, y, w in vm_examples(fr, CFG)] == [(0.0, 1.0), (0.0, sw), (1.0, sw), (0.0, sw)], \
        "метки VM: подзадача с ответом при нерешённой задаче должна быть «0» и идти в обучение"
    assert [is_sub_text(vm_text(f)) for f in fr[:2]] == [False, True], "уровень по тексту VM определяется неверно"
    st_ = {"reqs": [{"dest": "SUB", "text": "Assume X is a knight. Who is who? = X is a knight, Y is a knave", "result": "X is a knight, Y is a knight"},
                    {"dest": "LLM", "text": "Is Y a knight?", "result": "no"}], "answer": None}
    assert step_line(1, st_) == ("S1: SUB Assume X is a knight. Who is who? = X is a knight, Y is a knave ; LLM Is Y a knight?\n"
                                 "R1: X is a knight, Y is a knight ; no"), step_line(1, st_)
    old = {"id": 0, "level": 0, "parent": None, "request": "q", "plan": "p", "steps": [dict(st_, plan="p", subs_before=4)],
           "path": [], "subs_left": 3, "answer": None, "root_query": "q"}
    nf = normalize_frames([old], CFG)[0]
    assert nf["steps"][0]["reqs"][0]["text"] == "Assume X is a knight. Who is who?", "мусор в запросе не вычищен"
    assert " = " not in nf["steps"][0]["next_state"] and "\nR1: " in nf["steps"][0]["next_state"], nf["steps"][0]["next_state"]
    assert shot_state("S1: LLM 3 + 4 = 7 ; LLM 10 - 2 = 8\nS2: ANSWER x = y") == "S1: LLM 3 + 4 ; LLM 10 - 2\nR1: 7 ; 8\nS2: ANSWER x = y"
    assert clean_action("PLAN: p\nSUB: q? = fake ; SUB q2\nANSWER: a = b") == "PLAN: p\nSUB: q?\nANSWER: a = b"
    assert strip_echo("Is X a knight?", "Is X a knight? = yes") == "yes" and strip_echo("Is X a knight?", "yes") == "yes"
    assert strip_echo("Is X a knight?", "is x a knight?") == "?"
    pa = parse_action("PLAN: next case\nS2: SUB Assume A is a knave. Who is who? ; LLM Is B a knight?\nR2: A is a knave", 0, CFG)
    assert [(r["dest"], r["text"]) for r in pa[1]] == [("SUB", "Assume A is a knave. Who is who?"), ("LLM", "Is B a knight?")], pa[1]
    assert parse_action("S3: ANSWER A is a knight", 0, CFG)[2] == "A is a knight"
    assert [r["text"] for r in parse_action("SUB: q?\nS2: SUB q?\nSUB: q2?", 0, CFG)[1]] == ["q?", "q2?"], "повтор запроса"
    rng = random.Random(0)
    for _ in range(200):                              # пачки generate: каждая подсказка ровно один раз, лимиты соблюдены
        lens = [rng.randint(1, 3000) for _ in range(rng.randint(1, 300))]
        bs = length_batches(lens, 32, 50000)
        assert sorted(i for b in bs for i in b) == list(range(len(lens)))
        assert all(len(b) <= 32 and (len(b) == 1 or len(b) * max(lens[i] for i in b) <= 50000) for b in bs)
    # судья (заглушка) и пары: разный исход — по R; одинаковый — по вердиктам; разошедшиеся порядки и ничьи — не берутся
    ck = Config.dry(task="kk")
    set_task(ck)
    assert len(judge_check(ck)["judge_check"]) == 2
    t = make_tasks(1, 3, ck)[0]
    mk = lambda tid, a, r, x: {"id": 2 * tid + a, "level": 0, "parent": None, "request": t["query"],
                               "truth": t["truth"], "task_id": tid, "attempt": a, "R": r, "answer": x, "path": [],
                               "steps": [{"reqs": [], "answer": x}]}
    fa, fb, fc, fd, fe, ff = mk(0, 0, 1.0, "a"), mk(0, 1, 0.0, "b"), mk(1, 0, 0.0, "c"), mk(1, 1, 0.0, "d"), \
        mk(2, 0, 0.0, "e"), mk(2, 1, 0.0, "f")
    cache = {job_key(fc, fd, 0): {"v": "B", "c": 4}, job_key(fc, fd, 1): {"v": "A", "c": 2},
             job_key(fe, ff, 0): {"v": "A", "c": 5}, job_key(fe, ff, 1): {"v": "A", "c": 5}}
    pl, st = pair_labels([(fa, fb), (fc, fd), (fe, ff)], cache, ck)
    assert [(x[0], x[1], round(x[2], 3)) for x in pl] == [(vm_text(fa), vm_text(fb), 1.0), (vm_text(fd), vm_text(fc), 0.6)], pl
    assert st.get("orders_disagree_dropped") == 1, st
    assert pair_labels([(fa, fb), (fc, fd)], cache, replace(ck, vm_pairs="R"))[0] == pl[:1], "vm_pairs=R: только пары по R"
    # попарное обучение VM: у победителя метка GOOD → доля верно упорядоченных пар растёт
    torch.manual_seed(0)                              # детерминированная заглушка VM (иначе порог плавает)
    vm = make_vm(ck)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    txt = lambda good: " ".join(rng.choice(words) for _ in range(10)) + (" GOOD" if good else " xx")
    prs = [(txt(True), txt(False), 1.0, i) for i in range(160)]
    acc0 = vm_pair_acc(vm, prs)
    train_vm_pairs(vm, prs, [], replace(ck, vm_point_weight=0.0), epochs=3)
    acc1 = vm_pair_acc(vm, prs)
    assert acc1 > max(0.7, acc0), (acc0, acc1)
    set_task(CFG)
    class Stop(Exception):
        pass
    real, calls = run_episodes, [0]

    def flaky(*a, **k):                               # обрыв сессии на второй части датасета
        calls[0] += 1
        if calls[0] == 2:
            raise Stop()
        return real(*a, **k)
    globals()["run_episodes"] = flaky
    try:
        run(Config.dry(iterations=1), "dry_boot_resume")
    except Stop:
        pass
    finally:
        globals()["run_episodes"] = real
    Dr = RunDirs(os.path.join(BASE_DIR, "dry_boot_resume"))
    assert os.path.exists(os.path.join(Dr.dataset, "part_000.json.gz")) and \
        not os.path.exists(os.path.join(Dr.dataset, "part_001.json.gz")), "часть датасета не сохранилась"
    cfg_r = Config.dry(iterations=1)
    run(cfg_r, "dry_boot_resume")
    data = load_dataset(Dr, cfg_r)
    base = make_tasks(dataset_task_count(cfg_r), 1_000, cfg_r)
    tops = [f for f in data if f["level"] == 0]
    assert [(f["task_id"], f["attempt"]) for f in tops] == \
        [(i, a) for i in range(len(base)) for a in range(cfg_r.dataset_attempts)], "сбор после обрыва сбился"
    assert [f["request"] for f in tops] == [t["query"] for t in base for _ in range(cfg_r.dataset_attempts)]
    for i, f in enumerate(data):
        assert f["id"] == i, "id кадров сбились"
        if f["parent"] is not None:
            pid, si, j = f["parent"]
            assert data[pid]["steps"][si]["reqs"][j].get("child") == i, "ссылки родитель↔ребёнок сбились"
    sub = wm_subset(data, replace(cfg_r, wm_attempts=5))
    assert len(attempt_trees(sub)) == 5 and all(f["id"] == i for i, f in enumerate(sub))
    for f in sub:
        if f["parent"] is not None:
            pid, si, j = f["parent"]
            assert sub[pid]["steps"][si]["reqs"][j].get("child") == f["id"], "часть датасета: ссылки сбились"
    run(Config.dry(iterations=1), "dry_test")         # как будто сессия оборвалась после 1-й итерации
    res = run(Config.dry(), "dry_test")               # продолжение с «Drive»: слоты, оптимизаторы, метрики
    it_full, it_grpo = res["world_model_seed0"]["iters"], res["grpo_seed0"]["iters"]
    assert any(m["imag"].get("multi_player_steps", 0) > 0 for m in it_full), "игра шага (SCAR) не проверена"
    assert any(m["grpo"].get("grpo_items", 0) > 0 for m in it_grpo), "обновление GRPO не проверено"
    assert "win_info" in res["world_model_seed0"]["wm0"], "нет диагностики окна"
    assert [m["it"] for m in it_full] == [1, 2] and [m["it"] for m in it_grpo] == [1, 2], "продолжение сбилось"
    n_cf = 0                                          # контрфакт для полной коалиции = настоящий переход
    for f in load_dataset(RunDirs(os.path.join(BASE_DIR, "dry_test")), CFG):
        for t_, st_ in enumerate(f["steps"]):
            k = len(parse_action(st_["action"], f["level"], CFG)[3])
            assert coalition_next(f, t_, tuple(range(k)), k, CFG) == st_["next_state"], (f["request"], t_)
            n_cf += 1
    print(f"контрфакты: полная коалиция совпала с настоящим переходом во всех {n_cf} шагах")
    res_kk = run(Config.dry(task="kk"), "dry_test_kk")
    Dk = RunDirs(os.path.join(BASE_DIR, "dry_test_kk"))
    js = load_json(os.path.join(Dk.judge, "summary.json"))
    assert js and js["requests"] > 0 and js["pairs_for_VM"] > 0 and os.path.exists(os.path.join(Dk.judge, "verdicts.json"))
    assert res_kk["world_model_seed0"]["vm0"].get("pairs_acc") is not None, "попарная VM на предобучении не проверена"
    n_calls = js["requests"]
    run(Config.dry(task="kk"), "dry_test_kk")         # повторный запуск: судья не спрашивается заново (кэш на Drive)
    assert load_json(os.path.join(Dk.judge, "summary.json"))["requests"] == n_calls, "судья спрошен повторно"
    # шарды датасета (несколько GPU): каждый собирает свои части без судьи, основной запуск доделывает остальное
    cfg_s = Config.dry(task="kk", iterations=0)
    run(cfg_s, "dry_shard", shard=(0, 2))
    run(cfg_s, "dry_shard", shard=(1, 2))
    Ds = RunDirs(os.path.join(BASE_DIR, "dry_shard"))
    assert sorted(os.listdir(Ds.dataset)) == ["part_000.json.gz", "part_001.json.gz", "part_002.json.gz", "settings.json"], os.listdir(Ds.dataset)
    assert not os.path.exists(os.path.join(Ds.judge, "verdicts.json")), "шард не должен спрашивать судью"
    st_s = load_json(os.path.join(Ds.logs, "status.json"))
    assert st_s["done"] and "шард 1/2" in st_s["run"], st_s
    run(cfg_s, "dry_shard")                           # основной запуск: части готовы, судья и всё остальное — здесь
    assert load_json(os.path.join(Ds.judge, "summary.json"))["requests"] > 0
    # остановка по control.json: между итерациями, состояние целое, повторный запуск продолжает
    save_json({"stop": True}, os.path.join(BASE_DIR, "dry_test_kk", "logs", "control.json"))
    try:
        run(Config.dry(task="kk", iterations=3), "dry_test_kk")
        raise AssertionError("остановка по control.json не сработала")
    except StopRequested:
        pass
    assert load_json(os.path.join(BASE_DIR, "dry_test_kk", "logs", "status.json"))["stopped"]
    assert os.path.exists(os.path.join(BASE_DIR, "dry_test_kk", "logs", "control.done.json"))
    res = run(Config.dry(task="kk", iterations=3), "dry_test_kk")
    assert [m["it"] for m in res["world_model_seed0"]["iters"]] == [1, 2, 3], "продолжение после остановки сбилось"
    run(Config.dry(iterations=0), "dry_pre")          # предобучение отдельно, эксперимент — с его копии
    fork_run("dry_pre", "dry_exp")
    res = run(Config.dry(), "dry_exp")
    assert [m["it"] for m in res["world_model_seed0"]["iters"]] == [1, 2] and "eval_pi0" in res["grpo_seed0"], "форк сбился"
    for name in ("dry_test", "dry_test_kk", "dry_exp"):
        check_layout(os.path.join(BASE_DIR, name))
    print("раскладка папок прогона: OK (logs/ — только лёгкое, weights/ — веса)")


# %% [markdown]
# ## 13. Быстрая проверка (≈5–10 мин, один раз на прогон)
# Без долгого прогона: актор делает несколько шагов обучения, адаптер пишется на Drive и читается обратно — log π
# должны совпасть; тот же адаптер как ref (π0) и VM с головой — тоже; пробная генерация всеми способами, которыми
# пользуется прогон; модель мира (предобученная) учится копировать правдоподобные состояния в окно — в логе видно
# состояние и окно рядом; судья отвечает на одну очевидную пару (≈$0.0002; ключ — OPENROUTER_API_KEY в разделе 0).
# Если ячейка упала, полный прогон не стартует.

# %%
def cli_args():
    """Параметры запуска на сервере (в Colab их нет; переменные окружения не нужны):
    python3 sleepwalker.py [--tasks N] [--iterations K] [--stage all|dataset|pretrain|experiment] [--shard i/n]
                           [--arms full,grpo_text] [--gpu N] [--gpu-gb GB] [--state-dir DIR] [--status [-v]]
                           [--stop [RUN]] [--dry]"""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tasks", type=int), ap.add_argument("--iterations", type=int)
    ap.add_argument("--stage", default="all"), ap.add_argument("--shard")
    ap.add_argument("--arms"), ap.add_argument("--gpu"), ap.add_argument("--gpu-gb", type=float), ap.add_argument("--state-dir")
    ap.add_argument("--status", action="store_true"), ap.add_argument("--dry", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true"), ap.add_argument("--stop", nargs="?", const="", default=None)
    return ap.parse_known_args()[0]


def kk_config(tasks=0, iterations=10):
    """Единственный набор настроек — официальный K&K (Config.pro); tasks — сколько задач обучения в датасете
    (0 — все 6200), от него же — объём для модели мира и число частей датасета."""
    tasks = tasks or 0
    cfg = Config.pro(dataset_tasks=tasks, iterations=iterations)
    if tasks:
        att = tasks * cfg.dataset_attempts
        cfg = replace(cfg, dataset_parts=max(1, math.ceil(att / 520)), wm_attempts=min(cfg.wm_attempts, att),
                      bc_max_steps=min(cfg.bc_max_steps, 3 * att))
    return cfg


ARGS = cli_args()
NB_TASKS = 1000   # для ноутбука: sleepwalker.ipynb — 1000 задач (≈6–7 ч всё), sleepwalker_pro.ipynb — 0 = все 6200
IN_NOTEBOOK = "ipykernel" in sys.modules or "google.colab" in sys.modules
N_TASKS = ARGS.tasks if ARGS.tasks is not None else (NB_TASKS if IN_NOTEBOOK else 0)   # сервер без флага — все задачи
FULL_CFG = kk_config(N_TASKS, ARGS.iterations or 10)   # (имя TASKS занято реестром задач)
PRE_NAME = f"kk_{N_TASKS or 'all'}_pre_v7"          # новое имя версии кода: прежние папки несовместимы
FULL_NAME = PRE_NAME.replace("_pre_", "_exp_")   # другой эксперимент с того же предобучения — другое имя здесь
# На своём сервере (см. run_server.sh): --tasks N — задач в датасете (без флага — все); --iterations K; --stage all |
# dataset | pretrain | experiment; --shard i/n — собирать только части датасета k ≡ i (mod n) (несколько GPU
# параллельно); --arms full,grpo_text — какие плечи (на двух GPU — по плечу на карту); --gpu N — какая карта;
# --gpu-gb — память карты вручную; --state-dir ПАПКА — где хранить прогоны (по умолчанию ./sleepwalker_runs);
# --status [-v] — состояние прогонов (без --state-dir — всех запомненных папок); --stop [ПРОГОН] — остановить на
# ближайшей границе (без имени — все идущие в папке); --dry — проверка на заглушках.
if ARGS.status:
    if ARGS.state_dir:
        remember_state_dir(BASE_DIR)
    print_status([BASE_DIR] if ARGS.state_dir else None, ARGS.verbose)
    sys.exit(0)
if ARGS.stop is not None:
    request_stop(ARGS.stop or None)
    sys.exit(0)
FULL_CFG = fit_gpu(FULL_CFG, ARGS.gpu_gb)
STAGE_ = ARGS.stage
SHARD_ = tuple(int(x) for x in ARGS.shard.split("/")) if ARGS.shard else None
if ARGS.arms:
    FULL_CFG = replace(FULL_CFG, arms=tuple(ARGS.arms.split(",")))
if DRY_RUN:
    dry_check()
elif SHARD_:
    run(replace(FULL_CFG, iterations=0), PRE_NAME, shard=SHARD_)
elif STAGE_ in ("all", "pretrain", "experiment"):
    adapter_check(FULL_CFG, PRE_NAME)

# %% [markdown]
# ## 14. Предобучение (один раз; 1000 задач ≈3–4 ч, все 6200 ≈8–12 ч — оценка)
# Датасет: базовая модель решает задачи обучения по полному тексту, по 2 попытки (модель мира ещё не нужна); судья
# сравнивает пары попыток в фоне. На датасете — VM (попарно), метки RM, модель мира целиком (с RM), π0 обоих плеч и их
# оценка. Всё на Drive; готовое не пересчитывается — после обрыва снова Run all.

# %%
if not DRY_RUN and not SHARD_ and STAGE_ in ("all", "pretrain", "dataset"):
    PRE = run(replace(FULL_CFG, iterations=0, arms=() if STAGE_ == "dataset" else FULL_CFG.arms), PRE_NAME)

# %% [markdown]
# ## 15. Эксперимент: обучение с подкреплением (≈3–8 ч — оценка)
# Стартует с копии предобучения. На каждой итерации актор решает каждую задачу дважды, судья сравнивает пары (в фоне),
# VM учится на них попарно. Если сессия оборвётся — снова Run all: готовое подхватится с Drive.
# Итог — MyDrive/sleepwalker/<имя эксперимента>/logs/results.json и logs/log.txt; всё лёгкое для анализа — в logs/
# (её можно скачать целиком), веса — в weights/. Что где лежит — README.txt в папке прогона.

# %%
if not DRY_RUN and not SHARD_ and STAGE_ in ("all", "experiment"):
    fork_run(PRE_NAME, FULL_NAME)
    RESULTS = run(FULL_CFG, FULL_NAME)
