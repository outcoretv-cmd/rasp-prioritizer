"""RASP — Streamlit app v8: consent + analytics + feedback + safety."""
import uuid
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from rasp_core import RASPMethod

# ============ ОПЦИОНАЛЬНЫЕ ИМПОРТЫ ============
try:
    import streamlit_analytics2 as streamlit_analytics
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False

try:
    from streamlit_gsheets import GSheetsConnection
    HAS_GSHEETS = True
except ImportError:
    HAS_GSHEETS = False


def sheets_available():
    """Google Sheets доступен только если есть пакет и настроены секреты."""
    if not HAS_GSHEETS:
        return False
    try:
        return (
            "connections" in st.secrets
            and "gsheets" in st.secrets["connections"]
        )
    except Exception:
        return False


# ============ КОНСТАНТЫ ============
APP_VERSION = "v8"
st.set_page_config(page_title="RASP", layout="centered")
st.title("RASP")


def uid():
    return uuid.uuid4().hex[:8]


def probability_label(p):
    if p == 0:
        return "🔴 Я уверен, что нет"
    if p < 25:
        return "🟠 Скорее нет, чем да"
    if p < 50:
        return "🟡 Нет, но я не уверен в этом"
    if p == 50:
        return "⚪ Может да, а может нет"
    if p <= 75:
        return "🟡 Да, но я не уверен в этом"
    if p < 100:
        return "🟢 Скорее да, чем нет"
    return "🟢 Я уверен, что да"


def lam_zone(lam):
    if lam < 0.15:
        return "только возможности"
    if lam < 0.4:
        return "склонен к возможностям"
    if lam < 0.6:
        return "сбалансирован"
    if lam < 0.85:
        return "склонен к осторожности"
    return "только риски"


# ============ ИНИЦИАЛИЗАЦИЯ ============
def init_state():
    old_version = st.session_state.get("_app_version")
    old_consent = st.session_state.get("consent_given", False)

    if old_version != APP_VERSION:
        for k in list(st.session_state.keys()):
            del st.session_state[k]

    defaults = {
        "_app_version": APP_VERSION,
        "consent_given": old_consent,
        "step": 0,
        "deals": [],
        "risk_factors": [],
        "opp_factors": [],
        "deals_store": {},
        "risk_store": {},
        "opp_store": {},
        "risk_priorities": {},
        "opp_priorities": {},
        "prob_store": {},
        "P_risk": None,
        "P_opp": None,
        "lam": 0.5,
        "final_result": None,
        "app_rating": None,
        "app_comment": "",
        "session_id": str(uuid.uuid4()),
        "start_time": datetime.now().isoformat(),
        "_init_deals": False,
        "_init_rf": False,
        "_init_of": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def next_step():
    st.session_state.step += 1
    st.rerun()


def prev_step():
    st.session_state.step -= 1
    st.rerun()


# ============ ЭКРАН СОГЛАСИЯ ============
def render_consent():
    st.header("Добро пожаловать в RASP")
    st.markdown(
        """
**RASP — уникальная методика определения собственных приоритетов.**

Данная версия приложения работает только для ситуаций, в которых **нельзя отказаться от дела**.
Если ты хочешь решить, от какого дела отказаться в жизни — эта версия в этом не поможет.
Но если тебе просто хочется понять, с какой задачи стоит начать — это приложение для тебя.

Алгоритм **не поможет снизить деловую нагрузку** — он поможет только упорядочить задачи.

RASP пока применим только для **краткосрочного** использования, но в будущем станет лучше
и научится помогать расставлять приоритеты в долгосрочной перспективе и даже принимать
решения в системе **«выбор-отказ»**.
        """
    )

    st.divider()
    st.subheader("Перед началом работы")

    st.markdown(
        """
Для улучшения приложения мы собираем **обезличенную статистику**.

**Что собирается:**
- Твои действия в интерфейсе (нажатия кнопок, перемещение по шагам);
- Список твоих дел, факторов и оценок — **в обезличенном виде**;
- Твой отзыв об приложении (оценка и комментарий).

**С какой целью:**
- Понять, какие функции наиболее востребованы и удобны;
- Выявить «узкие места» в интерфейсе;
- Собрать предложения по улучшению.

**Твои данные обезличены.** Вместо имени и других личных данных используется
случайный идентификатор. Мы **не собираем** имена, e-mail, номера телефонов
или другие персональные сведения.

Нажимая «Согласен, продолжить», ты даёшь согласие на сбор и обработку
обезличенных данных в исследовательских целях.
        """
    )

    consent = st.checkbox(
        "Я согласен на сбор обезличенных данных и хочу продолжить",
        key="consent_checkbox",
    )
    if consent:
        st.session_state.consent_given = True
        st.rerun()


# ============ ДИНАМИЧЕСКИЕ ЯЧЕЙКИ ============
def render_cell_list(items_key, store_key, placeholder):
    items = st.session_state[items_key]
    store = st.session_state[store_key]

    current_ids = {x["id"] for x in items}
    for iid in list(store.keys()):
        if iid not in current_ids:
            del store[iid]

    to_delete = None
    for i, item in enumerate(items):
        c1, c2 = st.columns([10, 1])
        with c1:
            wkey = f"{items_key}_inp_{item['id']}"
            if wkey not in st.session_state:
                st.session_state[wkey] = store.get(item["id"], "")
            val = st.text_input(
                f"cell_{i}",
                key=wkey,
                placeholder=f"{placeholder} {i+1}",
                label_visibility="collapsed",
            )
            store[item["id"]] = val
        with c2:
            if st.button("🗑", key=f"{items_key}_del_{item['id']}"):
                to_delete = item["id"]

    if to_delete:
        st.session_state[items_key] = [
            x for x in items if x["id"] != to_delete
        ]
        store.pop(to_delete, None)
        st.rerun()

    if st.button("➕ Добавить", key=f"{items_key}_add"):
        st.session_state[items_key].append({"id": uid()})
        st.rerun()


def collect_items(items_key, store_key):
    store = st.session_state[store_key]
    result = []
    for item in st.session_state[items_key]:
        name = store.get(item["id"], "").strip()
        if name:
            result.append((item["id"], name))
    return result


# ============ СВЯЗАННЫЕ ПОЛЗУНКИ ============
def render_linked_priorities(items, values_key, prefix):
    if not items:
        return

    ids = [x[0] for x in items]

    if values_key not in st.session_state:
        st.session_state[values_key] = {}
    values = dict(st.session_state[values_key])

    for fid in list(values.keys()):
        if fid not in ids:
            del values[fid]

    if any(fid not in values for fid in ids):
        n = len(ids)
        base = 100 // n
        values = {fid: base for fid in ids}
        values[ids[0]] += 100 - sum(values.values())
        st.session_state[values_key] = values

    changed_id = None
    for fid in ids:
        wk = f"{prefix}_sl_{fid}"
        if wk in st.session_state and st.session_state[wk] != values[fid]:
            changed_id = fid
            break

    if changed_id:
        new_val = st.session_state[f"{prefix}_sl_{changed_id}"]
        values[changed_id] = new_val
        others = [x for x in ids if x != changed_id]
        remaining = 100 - new_val

        if others:
            old_others = {x: values.get(x, 0) for x in others}
            old_sum = sum(old_others.values())
            if old_sum > 0:
                new_others = {
                    x: int(round(old_others[x] * remaining / old_sum))
                    for x in others
                }
            else:
                eq = remaining // len(others)
                new_others = {x: eq for x in others}

            diff = remaining - sum(new_others.values())
            if diff != 0:
                largest = max(new_others, key=new_others.get)
                new_others[largest] += diff

            for x in others:
                values[x] = new_others[x]

        st.session_state[values_key] = values
        for fid in ids:
            if fid != changed_id:
                st.session_state[f"{prefix}_sl_{fid}"] = values[fid]
        st.rerun()
    else:
        for fid in ids:
            wk = f"{prefix}_sl_{fid}"
            if wk not in st.session_state:
                st.session_state[wk] = values[fid]

    for fid, name in items:
        c1, c2, c3 = st.columns([3, 5, 1])
        with c1:
            st.markdown(
                f"<div style='padding-top:10px'>{name}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.slider(
                name, 0, 100,
                key=f"{prefix}_sl_{fid}",
                label_visibility="collapsed",
            )
        with c3:
            st.markdown(
                f"<div style='padding-top:10px;text-align:right'>"
                f"{values[fid]}%</div>",
                unsafe_allow_html=True,
            )


def priorities_to_rasp(items, priorities):
    if not items:
        return [], []
    sorted_items = sorted(items, key=lambda x: -priorities.get(x[0], 0))
    values = [max(1, priorities.get(fid, 0)) for fid, _ in sorted_items]
    q = [values[j] / values[j + 1] for j in range(len(values) - 1)]
    return [name for _, name in sorted_items], q


# ============ ШАГИ ============
def step_deals():
    st.subheader("Шаг 1. Список дел")
    st.write("Добавь дела, между которыми нужно расставить приоритеты.")

    if not st.session_state._init_deals:
        st.session_state.deals = [{"id": uid()} for _ in range(2)]
        st.session_state._init_deals = True
        st.rerun()

    render_cell_list("deals", "deals_store", "Дело")

    st.divider()
    if st.button("Далее →", type="primary", key="next_deals"):
        names = collect_items("deals", "deals_store")
        if len(names) < 2:
            st.error("Нужно хотя бы два дела.")
            return
        next_step()


def step_risk():
    st.header("Шаг 2. Факторы риска")
    st.write(
        "Риск — это то, чего ты хочешь избежать. "
        "Перечисли неприятности, с которыми ты можешь столкнуться, "
        "если **не сделаешь** свои дела."
    )

    if not st.session_state._init_rf:
        st.session_state.risk_factors = [{"id": uid()} for _ in range(3)]
        st.session_state._init_rf = True
        st.rerun()

    render_cell_list("risk_factors", "risk_store", "Фактор")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_risk"):
            prev_step()
    with c2:
        if st.button("Далее →", type="primary", key="next_risk"):
            names = collect_items("risk_factors", "risk_store")
            if len(names) < 1:
                st.error("Добавь хотя бы один фактор.")
                return
            next_step()


def step_opp():
    st.header("Шаг 3. Факторы возможностей")
    st.write(
        "Возможности — это выгода, которую ты хочешь получить. "
        "Перечисли бонусы, которые ты получишь, если **сделаешь** свои дела."
    )

    if not st.session_state._init_of:
        st.session_state.opp_factors = [{"id": uid()} for _ in range(3)]
        st.session_state._init_of = True
        st.rerun()

    render_cell_list("opp_factors", "opp_store", "Возможность")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_opp"):
            prev_step()
    with c2:
        if st.button("Далее →", type="primary", key="next_opp"):
            names = collect_items("opp_factors", "opp_store")
            if len(names) < 1:
                st.error("Добавь хотя бы один фактор.")
                return
            next_step()


def step_risk_priority():
    st.header("Шаг 4. Приоритет рисков")
    st.write(
        "Двигай ползунки, чтобы их расположение соответствовало "
        "твоим представлениям о важности факторов риска. Ползунки "
        "**взаимосвязаны**: на одном месте не могут быть 2 фактора — "
        "придётся выбирать, какой важнее. Самый важный фактор имеет "
        "самое большое значение, наименее важный — самое низкое."
    )

    items = collect_items("risk_factors", "risk_store")
    if not items:
        st.error("Вернись на шаг 2 и заполни факторы риска.")
        if st.button("← Назад"):
            prev_step()
        return

    render_linked_priorities(items, "risk_priorities", "rprio")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_rprio"):
            prev_step()
    with c2:
        if st.button("Далее →", type="primary", key="next_rprio"):
            next_step()


def step_opp_priority():
    st.header("Шаг 5. Приоритет возможностей")
    st.write(
        "Двигай ползунки, чтобы их расположение соответствовало "
        "твоим представлениям о важности факторов возможностей. "
        "Ползунки **взаимосвязаны**: на одном месте не могут быть "
        "2 фактора — придётся выбирать, какой важнее."
    )

    items = collect_items("opp_factors", "opp_store")
    if not items:
        st.error("Вернись на шаг 3 и заполни факторы возможностей.")
        if st.button("← Назад"):
            prev_step()
        return

    render_linked_priorities(items, "opp_priorities", "oprio")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_oprio"):
            prev_step()
    with c2:
        if st.button("Далее →", type="primary", key="next_oprio"):
            next_step()


def step_probabilities():
    st.header("Шаг 6. Вероятности")

    with st.expander("❓ Как понимать ползунки", expanded=False):
        st.markdown("""
**Шкала от 0 до 100.** Чем ближе к 50 — тем меньше уверенности.

| Значение | Что значит |
|:---:|---|
| 0 | 🔴 Я уверен, что нет |
| 1–24 | 🟠 Скорее нет, чем да |
| 25–49 | 🟡 Нет, но я не уверен |
| 50 | ⚪ Может да, а может нет |
| 51–75 | 🟡 Да, но я не уверен |
| 76–99 | 🟢 Скорее да, чем нет |
| 100 | 🟢 Я уверен, что да |

- **Риски:** вероятность наступления, если дело **НЕ сделать**.
- **Возможности:** вероятность реализации, если дело **сделать**.
        """)

    deals = collect_items("deals", "deals_store")
    r_items = collect_items("risk_factors", "risk_store")
    o_items = collect_items("opp_factors", "opp_store")

    r_sorted = sorted(
        r_items, key=lambda x: -st.session_state.risk_priorities.get(x[0], 0)
    )
    o_sorted = sorted(
        o_items, key=lambda x: -st.session_state.opp_priorities.get(x[0], 0)
    )

    prob_store = st.session_state.prob_store

    st.subheader("Риски")
    st.caption("Вероятность, что риск наступит, если дело **не сделать**.")
    P_risk = np.zeros((len(deals), len(r_sorted)))
    for i, (deal_id, deal_name) in enumerate(deals):
        with st.container(border=True):
            st.markdown(f"**{deal_name}**")
            for j, (fid, fname) in enumerate(r_sorted):
                key = f"pr_{deal_id}_{fid}"
                if key not in st.session_state:
                    st.session_state[key] = prob_store.get(key, 50)
                val = st.slider(fname, 0, 100, step=5, key=key)
                st.caption(probability_label(val))
                P_risk[i, j] = val / 100.0
                prob_store[key] = val

    st.subheader("Возможности")
    st.caption("Вероятность, что возможность реализуется, если дело **сделать**.")
    P_opp = np.zeros((len(deals), len(o_sorted)))
    for i, (deal_id, deal_name) in enumerate(deals):
        with st.container(border=True):
            st.markdown(f"**{deal_name}**")
            for j, (fid, fname) in enumerate(o_sorted):
                key = f"po_{deal_id}_{fid}"
                if key not in st.session_state:
                    st.session_state[key] = prob_store.get(key, 50)
                val = st.slider(fname, 0, 100, step=5, key=key)
                st.caption(probability_label(val))
                P_opp[i, j] = val / 100.0
                prob_store[key] = val

    st.session_state.P_risk = P_risk
    st.session_state.P_opp = P_opp

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_prob"):
            prev_step()
    with c2:
        if st.button("Далее →", type="primary", key="next_prob"):
            next_step()


def step_lambda():
    st.header("Шаг 7. Отношение к риску")
    st.write(
        "Подумай, что для тебя важнее — **избегать рисков** или "
        "**получать возможности**? Отметь ползунком на шкале, "
        "насколько ты готов рисковать ради получения возможностей."
    )

    lam_key = "lam_slider"
    if lam_key not in st.session_state:
        st.session_state[lam_key] = float(st.session_state.lam)

    c1, c2, c3 = st.columns([1, 5, 1])
    with c1:
        st.markdown(
            "<div style='padding-top:10px;text-align:left;font-size:13px'>"
            "Склонен<br>к возможностям</div>",
            unsafe_allow_html=True,
        )
    with c2:
        lam = st.slider(
            "λ",
            min_value=0.0, max_value=1.0,
            step=0.05,
            key=lam_key,
            label_visibility="collapsed",
        )
    with c3:
        st.markdown(
            "<div style='padding-top:10px;text-align:right;font-size:13px'>"
            "Склонен<br>к избеганию рисков</div>",
            unsafe_allow_html=True,
        )

    st.session_state.lam = lam
    st.info(f"λ = {lam:.2f} — {lam_zone(lam)}")

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_lam"):
            prev_step()
    with c2:
        if st.button("Рассчитать →", type="primary", key="next_lam"):
            next_step()


def step_result():
    st.header("Шаг 8. Результат")

    deals = collect_items("deals", "deals_store")
    deal_names = [n for _, n in deals]
    r_items = collect_items("risk_factors", "risk_store")
    o_items = collect_items("opp_factors", "opp_store")

    r_names, q_risk = priorities_to_rasp(r_items, st.session_state.risk_priorities)
    o_names, q_opp = priorities_to_rasp(o_items, st.session_state.opp_priorities)

    m = RASPMethod(q_risk=q_risk, q_opp=q_opp)
    res = m.rank(
        st.session_state.P_risk,
        st.session_state.P_opp,
        lam=st.session_state.lam,
        deal_names=deal_names,
    )

    # Сохраняем финальный рейтинг для отчёта
    st.session_state.final_result = [
        {"rank": r + 1, "deal": res.deal_names[i], "V": float(res.V[i])}
        for r, i in enumerate(res.order)
    ]

    # -------- Основной рейтинг --------
    st.subheader("Рейтинг")
    for rank, i in enumerate(res.order, start=1):
        st.markdown(
            f"**{rank}. {res.deal_names[i]}** — V = {res.V[i]:.3f} "
            f"(R = {res.R[i]:.3f}, O = {res.O[i]:.3f})"
        )

    st.subheader("Веса факторов")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Риски**")
        for f, w in zip(r_names, m.w_risk):
            st.write(f"- {f}: {w*100:.1f}%")
    with c2:
        st.write("**Возможности**")
        for f, w in zip(o_names, m.w_opp):
            st.write(f"- {f}: {w*100:.1f}%")

    chart = {res.deal_names[i]: float(res.V[i]) for i in res.order}
    st.bar_chart(chart)

    # -------- Устойчивость --------
    st.divider()
    st.subheader("Насколько можно доверять этому рейтингу?")
    st.write(
        "Рейтинг зависит от твоего отношения к риску (λ). "
        "Мы проверили, что произойдёт, если менять этот настрой от 0 "
        "(только возможности) до 1 (только риски)."
    )

    lambdas = [round(x * 0.05, 2) for x in range(21)]
    V_by_lam = {name: [] for name in deal_names}
    leader_by_lam = []
    margin_by_lam = []

    for lam in lambdas:
        r = m.rank(
            st.session_state.P_risk,
            st.session_state.P_opp,
            lam=lam,
            deal_names=deal_names,
        )
        for i, name in enumerate(deal_names):
            V_by_lam[name].append(float(r.V[i]))
        leader_by_lam.append(r.ranked_deals()[0])
        sorted_v = sorted(r.V, reverse=True)
        margin = float(sorted_v[0] - sorted_v[1]) if len(sorted_v) > 1 else 0.0
        margin_by_lam.append(margin)

    current_lam = st.session_state.lam
    closest_lam = min(lambdas, key=lambda x: abs(x - current_lam))
    current_idx = lambdas.index(closest_lam)
    current_leader = leader_by_lam[current_idx]
    current_margin = margin_by_lam[current_idx]

    n_switches = sum(
        1 for j in range(1, len(leader_by_lam))
        if leader_by_lam[j] != leader_by_lam[j - 1]
    )

    def steps_to_switch(idx):
        for d in range(1, len(leader_by_lam)):
            if idx - d >= 0 and leader_by_lam[idx - d] != leader_by_lam[idx]:
                return d
            if idx + d < len(leader_by_lam) and leader_by_lam[idx + d] != leader_by_lam[idx]:
                return d
        return None

    steps = steps_to_switch(current_idx)

    if n_switches == 0:
        if current_margin >= 0.10:
            st.success(
                f"✅ **Надёжное решение.** При любом отношении к риску "
                f"лидирует **{current_leader}**. Отрыв — "
                f"{current_margin:.2f} балла."
            )
        else:
            st.warning(
                f"⚠️ **Умеренно устойчивое.** Лидер один и тот же — "
                f"**{current_leader}**, но отрыв всего "
                f"{current_margin:.2f} балла. Дела почти равнозначны."
            )
    else:
        if steps is not None and steps == 1:
            st.error(
                f"❗ **Решение на грани.** Ты в точке λ = {current_lam:.2f} "
                f"(«{lam_zone(current_lam)}»), и всего один шаг до смены "
                f"лидера. Малейшее изменение настроя перевернёт рейтинг."
            )
        elif steps is not None and steps <= 3:
            st.warning(
                f"⚠️ **Умеренно устойчивое.** Ты в точке λ = "
                f"{current_lam:.2f} («{lam_zone(current_lam)}»), лидирует "
                f"**{current_leader}**. До смены — {steps} шага."
            )
        else:
            st.success(
                f"✅ **Надёжное для твоего настроя.** При λ = "
                f"{current_lam:.2f} («{lam_zone(current_lam)}») лидирует "
                f"**{current_leader}** с отрывом {current_margin:.2f}."
            )

    st.write("**Кто выходит в лидеры при разных λ:**")
    ranges = []
    current_l = leader_by_lam[0]
    start_lam = lambdas[0]
    prev_lam = lambdas[0]
    for lam, leader in zip(lambdas[1:], leader_by_lam[1:]):
        if leader != current_l:
            ranges.append((current_l, start_lam, prev_lam))
            current_l = leader
            start_lam = lam
        prev_lam = lam
    ranges.append((current_l, start_lam, lambdas[-1]))

    for leader, lo, hi in ranges:
        if lo == hi:
            st.write(f"- **{leader}** — при λ = {lo:.2f}")
        else:
            st.write(
                f"- **{leader}** — от «{lam_zone(lo)}» до "
                f"«{lam_zone(hi)}» (λ от {lo:.2f} до {hi:.2f})"
            )

    if n_switches == 0:
        st.info(
            "💡 **Почему лидер не меняется.** Одно дело доминирует: "
            "у него и риски выше, и возможности выше остальных."
        )

    with st.expander("🔍 Диагностика вердикта", expanded=False):
        st.write(f"**Текущая λ:** {current_lam:.2f}")
        st.write(f"**Лидер:** {current_leader}")
        st.write(f"**Отрыв от второго места:** {current_margin:.3f}")
        st.write(f"**Число смен лидера:** {n_switches}")
        st.write(f"**Шагов до смены:** {steps if steps is not None else '—'}")

    st.write("**Как меняется балл каждого дела:**")
    st.caption("Точки пересечения линий — моменты смены лидера.")
    line_data = {"λ": lambdas}
    for name in deal_names:
        line_data[name] = V_by_lam[name]
    st.line_chart(line_data, x="λ")

    # -------- Обратная связь --------
    st.divider()
    st.subheader("Поделитесь мнением")

    rating = st.feedback("stars", key="app_rating_widget")
    if rating is not None:
        st.session_state.app_rating = rating

    comment = st.text_area(
        "Что можно улучшить? (необязательно)",
        value=st.session_state.app_comment,
        key="app_comment_widget",
    )
    st.session_state.app_comment = comment

    if st.button("Отправить отзыв", key="submit_feedback"):
        if st.session_state.app_rating is None and not comment.strip():
            st.warning("Поставь оценку или напиши комментарий.")
        elif not sheets_available():
            st.info(
                "Сбор отзывов временно недоступен (не настроены секреты). "
                "Спасибо за прохождение!"
            )
        else:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                existing = conn.read(worksheet="feedback", ttl=0)
                new_row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "session_id": st.session_state.session_id,
                    "rating": (
                        st.session_state.app_rating + 1
                        if st.session_state.app_rating is not None else None
                    ),
                    "comment": comment,
                }])
                updated = pd.concat([existing, new_row], ignore_index=True)
                conn.update(worksheet="feedback", data=updated)
                st.success("Спасибо! Отзыв отправлен.")
            except Exception as e:
                st.error(f"Не удалось отправить отзыв: {e}")

    # -------- Сохранение сессии --------
    if st.button("Завершить и сохранить сессию", key="save_session"):
        session_data = {
            "session_id": st.session_state.session_id,
            "start_time": st.session_state.start_time,
            "end_time": datetime.now().isoformat(),
            "deals": list(st.session_state.deals_store.values()),
            "risk_factors": list(st.session_state.risk_store.values()),
            "opp_factors": list(st.session_state.opp_store.values()),
            "risk_priorities": st.session_state.risk_priorities,
            "opp_priorities": st.session_state.opp_priorities,
            "lambda": st.session_state.lam,
            "final_ranking": st.session_state.final_result,
            "app_rating": st.session_state.app_rating,
            "app_comment": st.session_state.app_comment,
        }

        if not sheets_available():
            st.info(
                "Сохранение сессий временно недоступно "
                "(не настроены секреты). Спасибо за прохождение!"
            )
        else:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                existing = conn.read(worksheet="sessions", ttl=0)
                new_row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "session_id": st.session_state.session_id,
                    "data": json.dumps(session_data, ensure_ascii=False),
                }])
                updated = pd.concat([existing, new_row], ignore_index=True)
                conn.update(worksheet="sessions", data=updated)
                st.success("Сессия сохранена. Спасибо за участие!")
            except Exception as e:
                st.error(f"Не удалось сохранить сессию: {e}")

    # -------- Навигация --------
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Назад", key="back_result"):
            prev_step()
    with c2:
        if st.button("Начать заново", key="restart"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ============ РОУТЕР ============
def render_main_app():
    steps = [
        step_deals,
        step_risk,
        step_opp,
        step_risk_priority,
        step_opp_priority,
        step_probabilities,
        step_lambda,
        step_result,
    ]

    progress = st.session_state.step / (len(steps) - 1)
    st.progress(progress)
    steps[st.session_state.step]()


# ============ ТОЧКА ВХОДА ============
if not st.session_state.consent_given:
    render_consent()
    st.stop()

# Аналитика — опциональна, при ошибке продолжит без неё
if HAS_ANALYTICS:
    try:
        with streamlit_analytics.track(
            session_id=st.session_state.session_id,
        ):
            render_main_app()
    except Exception:
        render_main_app()
else:
    render_main_app()
