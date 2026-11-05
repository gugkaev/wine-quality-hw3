"""Демонстрация сохранённой модели отбора вин; обучение выполняется в ноутбуке.

Кэширование (задание 4.5): st.cache_resource сохраняет один объект модели
для всех перезапусков и сессий. Модель в приложении только читает данные.
st.cache_data подходит для таблиц и массивов: результат сериализуется,
а вызывающий код получает собственную копию. Без декоратора joblib.load
снова читал бы файл при каждом изменении виджета. Версии файлов входят
в ключи кэша, поэтому замена model.joblib/test.csv обновляет результаты.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
WINE_COLOR = "#87304d"
GOOD_COLOR = "#23775d"

# Подписи не меняют имена и порядок столбцов, ожидаемых моделью.
FEATURE_LABELS = {
    "fixed_acidity": "Нелетучая кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "is_red": "Красное вино",
}
INPUT_STEPS = {
    "fixed_acidity": 0.1,
    "volatile_acidity": 0.01,
    "citric_acid": 0.01,
    "residual_sugar": 0.1,
    "chlorides": 0.001,
    "free_sulfur_dioxide": 1.0,
    "total_sulfur_dioxide": 1.0,
    "density": 0.00001,
    "pH": 0.01,
    "sulphates": 0.01,
    "alcohol": 0.1,
}


def file_signature(path: Path) -> tuple[str, int, int]:
    """Версия локального файла для обновления кэша после его замены."""
    info = path.stat()
    return str(path), info.st_mtime_ns, info.st_size


@st.cache_resource(show_spinner="Загружаем модель…")
def load_model(path: str, version: tuple[str, int, int]):
    """Один неизменяемый в приложении объект обученного пайплайна."""
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_test_data(path: str, version: tuple[str, int, int]) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def test_proba(_model, X: pd.DataFrame, model_version: tuple[str, int, int]):
    """Порог не входит в ключ: его сдвиг не запускает predict_proba заново."""
    return np.asarray(_model.predict_proba(X)[:, 1], dtype=float)


@st.cache_data(show_spinner=False, max_entries=64)
def other_proba(_model, X: pd.DataFrame, model_version: tuple[str, int, int]):
    """Также кешируем прогнозы одного вина и загруженной партии."""
    return np.asarray(_model.predict_proba(X)[:, 1], dtype=float)


def validate_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Возвращает числовые признаки в порядке обучения или понятную ошибку."""
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError("В CSV не хватает столбцов: " + ", ".join(missing) + ".")
    if frame.empty:
        raise ValueError("В CSV есть заголовок, но нет ни одной строки с вином.")
    if frame.columns.duplicated().any():
        raise ValueError("Имена столбцов в CSV не должны повторяться.")

    X = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(X.to_numpy(dtype=float))
    if not finite.all():
        invalid_columns = X.columns[~finite.all(axis=0)].tolist()
        raise ValueError(
            "Нужны конечные числа без пропусков. Проверьте столбцы: "
            + ", ".join(invalid_columns)
            + ". Десятичный разделитель — точка."
        )
    if "is_red" in X and not X["is_red"].isin([0, 1]).all():
        raise ValueError("Столбец is_red должен содержать только 0 (белое) или 1 (красное).")
    return X


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 BOM сохраняет русские подписи при открытии CSV в Excel."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def reset_threshold(value: float) -> None:
    st.session_state["threshold"] = value


def render_single_wine(model, X_test, features, threshold, model_version) -> None:
    st.subheader("Оценка одного вина")
    st.write("Введите результаты лабораторного анализа. Модель оценивает вероятность экспертной оценки не ниже 7 из 10.")
    st.caption("Начальные значения — медианы; допустимые диапазоны взяты из test.csv.")

    is_red = st.toggle(
        "Красное вино",
        value=bool(X_test["is_red"].median() >= 0.5),
        key="wine_is_red",
        help="Включено — красное; выключено — белое.",
    )
    st.caption("Тип: красное" if is_red else "Тип: белое")
    values = {"is_red": int(is_red)}
    columns = st.columns(3)
    numeric_features = [name for name in features if name != "is_red"]
    for i, name in enumerate(numeric_features):
        low = float(X_test[name].min())
        high = float(X_test[name].max())
        median = float(X_test[name].median())
        number_format = "%.5f" if name == "density" else "%.3f" if name == "chlorides" else "%.2f"
        with columns[i % 3]:
            values[name] = st.number_input(
                FEATURE_LABELS.get(name, name),
                min_value=low,
                max_value=high,
                value=median,
                step=INPUT_STEPS.get(name, 0.01),
                format=number_format,
                key=f"wine_{name}",
                help=f"Столбец {name}; диапазон {low:g}–{high:g}.",
            )

    one_wine = pd.DataFrame([values], columns=features)
    probability = float(other_proba(model, one_wine, model_version)[0])
    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Вероятность хорошего вина", f"{probability:.1%}")
    with right:
        if probability >= threshold:
            st.success("Решение: включить в подборку")
        else:
            st.info("Решение: не включать в подборку")
        st.caption(f"Правило: вероятность ≥ порога. Сейчас p = {probability:.6f}, порог = {threshold:.6f}.")


def render_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    st.subheader("Модельные метрики")
    st.write(
        "Более высокий порог делает отбор строже: рекомендаций становится меньше, "
        "а часть хороших вин теряется. Более низкий порог расширяет подборку и повышает "
        "полноту, но в неё могут попадать посредственные вина. Precision обычно растёт "
        "при повышении порога, однако на конечной выборке это не гарантировано."
    )
    st.caption(
        "Порог по умолчанию выбран на валидации. Здесь показана финальная тестовая выборка: "
        "слайдер иллюстрирует компромисс, а новый рабочий порог следует выбирать на валидации."
    )
    predicted = (scores >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_true, predicted, zero_division=0)
    recall = metrics.recall_score(y_true, predicted, zero_division=0)
    f1 = metrics.f1_score(y_true, predicted, zero_division=0)
    n_selected = int(predicted.sum())

    cards = st.columns(4)
    cards[0].metric("Precision", f"{precision:.3f}", help="Доля хороших вин среди рекомендованных. Если рекомендаций нет, отображается 0.")
    cards[1].metric("Recall", f"{recall:.3f}", help="Доля найденных хороших вин среди всех хороших вин.")
    cards[2].metric("F1", f"{f1:.3f}")
    cards[3].metric("В подборке", f"{n_selected} / {len(y_true)}")

    matrix_column, explanation_column = st.columns([1, 1])
    with matrix_column:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        metrics.ConfusionMatrixDisplay(cm, display_labels=["Остальные", "Хорошие"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_xlabel("Решение модели")
        ax.set_ylabel("Экспертная оценка")
        ax.set_title("Матрица ошибок")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with explanation_column:
        st.markdown(
            f"**TP = {tp}:** хорошие вина, включённые в подборку.\n\n"
            f"**FP = {fp}:** посредственные вина, включённые в подборку.\n\n"
            f"**FN = {fn}:** хорошие вина, которые модель пропустила.\n\n"
            f"**TN = {tn}:** остальные вина, которые модель отклонила."
        )
        st.metric("Accuracy", f"{metrics.accuracy_score(y_true, predicted):.3f}")
        st.caption("Из-за дисбаланса классов accuracy следует оценивать вместе с precision и recall.")

    fpr, tpr, _ = metrics.roc_curve(y_true, scores)
    curve_precision, curve_recall, _ = metrics.precision_recall_curve(y_true, scores)
    roc_auc = metrics.roc_auc_score(y_true, scores)
    average_precision = metrics.average_precision_score(y_true, scores)
    prevalence = float(np.mean(y_true))
    current_fpr = fp / (fp + tn) if fp + tn else 0.0
    label = f"Порог {threshold:.4f}"
    roc_column, pr_column = st.columns(2)

    with roc_column:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.plot(fpr, tpr, color=WINE_COLOR, label=f"ROC-AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Случайное ранжирование")
        ax.scatter([current_fpr], [recall], color=GOOD_COLOR, s=70, zorder=4, label=label)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Доля ложных срабатываний (FPR)", ylabel="Полнота (TPR)", title="ROC-кривая")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with pr_column:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.step(curve_recall, curve_precision, where="post", color=WINE_COLOR, label=f"PR-AUC (AP) = {average_precision:.3f}")
        ax.axhline(prevalence, linestyle="--", color="#9ca3af", label=f"Случайная модель: {prevalence:.3f}")
        # При отсутствии положительных прогнозов sklearn замыкает PR-кривую
        # условной точкой (recall=0, precision=1). В таблице zero_division=0.
        pr_point_precision = precision if n_selected else 1.0
        ax.scatter([recall], [pr_point_precision], color=GOOD_COLOR, s=70, zorder=4, label=label)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Полнота (Recall)", ylabel="Точность (Precision)", title="PR-кривая")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.caption("ROC-AUC и PR-AUC (average precision) используют вероятности и не меняются при сдвиге порога.")
    if not n_selected:
        st.info(
            "При этом пороге рекомендаций нет, поэтому precision не определён. "
            "В карточке он показан как 0; на PR-кривой используется условная конечная точка (0, 1), как в sklearn."
        )


def render_batch(model, X_test, features, threshold, model_version) -> None:
    st.subheader("Пакетное предсказание")
    st.write("Загрузите CSV с 12 признаками: одна строка — одно вино. Заголовки должны совпадать с примером; target не нужен.")
    st.caption(f"Для всей партии используется текущий общий порог: {threshold:.6f}.")
    st.download_button(
        "Скачать пример CSV",
        data=csv_bytes(X_test.loc[:, features].head(5)),
        file_name="wine_example.csv",
        mime="text/csv",
        key="download_example",
    )
    uploaded = st.file_uploader("CSV с признаками вин", type=["csv"], key="wine_csv")
    if uploaded is None:
        with st.expander("Ожидаемые названия столбцов"):
            st.code(",".join(features), language="text")
        return

    try:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=None, engine="python")
        batch_X = validate_features(frame, features)
    except (csv.Error, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Не удалось обработать CSV. {exc}")
        return

    extra_columns = [name for name in frame.columns if name not in features]
    if extra_columns:
        st.caption("Дополнительные столбцы сохранены в результате, но не используются моделью: " + ", ".join(extra_columns) + ".")
    probabilities = other_proba(model, batch_X, model_version)
    decisions = (probabilities >= threshold).astype(int)
    result = frame.copy()
    result["probability_good"] = probabilities
    result["prediction"] = decisions
    result["decision"] = np.where(decisions == 1, "В подборку", "Не включать")

    st.success(f"Обработано вин: {len(result)}. В подборку включено: {int(decisions.sum())}.")
    st.dataframe(
        result,
        width="stretch",
        hide_index=True,
        column_config={
            "probability_good": st.column_config.NumberColumn("P(хорошее вино)", format="%.4f"),
            "prediction": st.column_config.NumberColumn("Класс (0/1)", format="%d"),
            "decision": st.column_config.TextColumn("Решение"),
        },
    )
    st.download_button(
        "Скачать результаты CSV",
        data=csv_bytes(result),
        file_name="wine_predictions.csv",
        mime="text/csv",
        key="download_predictions",
    )


def render_coefficients(model, features: list[str]) -> None:
    st.subheader("Что выучила модель")
    classifier = model.steps[-1][1]
    coefficients = pd.Series(classifier.coef_[0], index=features).sort_values()
    colors = np.where(coefficients >= 0, GOOD_COLOR, WINE_COLOR)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh([FEATURE_LABELS.get(name, name) for name in coefficients.index], coefficients, color=colors)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Коэффициент после StandardScaler")
    ax.set_title("Вклад признаков в логарифм шансов хорошего вина")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    positive_text = ", ".join(f"{FEATURE_LABELS.get(name, name)} ({value:+.3f})" for name, value in positive.items()) or "положительных коэффициентов нет"
    negative_text = ", ".join(f"{FEATURE_LABELS.get(name, name)} ({value:+.3f})" for name, value in negative.items()) or "отрицательных коэффициентов нет"
    st.markdown(f"**Сильнее всего за хорошее вино:** {positive_text}.\n\n**Сильнее всего против:** {negative_text}.")
    st.caption(
        "При прочих равных увеличение стандартизированного признака на 1 меняет логарифм шансов "
        "на величину его коэффициента. Коэффициенты описывают связи в данных; корреляция признаков "
        "влияет на их значения, а причинность из них не следует."
    )
    with st.expander("Таблица коэффициентов"):
        st.dataframe(
            coefficients.rename("coefficient").rename_axis("feature").reset_index(),
            width="stretch",
            hide_index=True,
        )
        st.write(f"Свободный член: {float(classifier.intercept_[0]):.6f}")


def main() -> None:
    st.set_page_config(page_title="Вина для подборки", page_icon="🍷", layout="wide")
    st.title("🍷 Вина для подборки")
    st.caption("Лабораторный анализ → оценка качества → понятный отбор для сомелье")

    model_path = APP_DIR / "model.joblib"
    config_path = APP_DIR / "config.json"
    test_path = APP_DIR / "test.csv"
    missing_files = [path.name for path in [model_path, config_path, test_path] if not path.is_file()]
    if missing_files:
        st.error("Рядом с app.py не найдены файлы: " + ", ".join(missing_files) + ". Сначала выполните ячейку сохранения модели в ноутбуке.")
        st.stop()

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        default_threshold = float(config["threshold"])
        if not np.isfinite(default_threshold) or not 0 <= default_threshold <= 1:
            raise ValueError("threshold в config.json должен быть числом от 0 до 1.")
        model_version = file_signature(model_path)
        model = load_model(str(model_path), model_version)
        test = load_test_data(str(test_path), file_signature(test_path))
        if "target" not in test:
            raise ValueError("В test.csv отсутствует target.")
        features = list(getattr(model, "feature_names_in_", config.get("features", test.drop(columns="target").columns.tolist())))
        X_test = validate_features(test, features)
        y_test = test["target"].to_numpy(dtype=int)
        if set(np.unique(y_test)) != {0, 1}:
            raise ValueError("Для расчёта ROC/PR в test.csv нужны оба класса target: 0 и 1.")
        scores = test_proba(model, X_test, model_version)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(f"Не удалось загрузить данные приложения: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Настройка отбора")
        if "threshold" not in st.session_state:
            st.session_state["threshold"] = default_threshold
        threshold = st.slider(
            "Порог хорошего вина",
            min_value=0.0,
            max_value=1.0,
            step=0.0001,
            format="%.4f",
            key="threshold",
            help="Вино попадает в подборку, если вероятность хорошего качества не ниже порога.",
        )
        st.caption(f"Порог из валидации: {default_threshold:.6f}")
        st.button("Вернуть исходный порог", on_click=reset_threshold, args=(default_threshold,))
        st.divider()
        st.metric("Хороших вин в тесте", f"{np.mean(y_test):.1%}")
        st.caption(f"Всего в тестовой выборке: {len(test)}.\n\nОдин порог применяется на всех вкладках.")

    single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(
        ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
    )
    with single_tab:
        render_single_wine(model, X_test, features, threshold, model_version)
    with metrics_tab:
        render_metrics(y_test, scores, threshold)
    with batch_tab:
        render_batch(model, X_test, features, threshold, model_version)
    with coefficients_tab:
        render_coefficients(model, features)


if __name__ == "__main__":
    main()
