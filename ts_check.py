import pandas as pd
import pickle
from config import PROCESSED_DATA_PATH, TEST_START_DATE

def analyze_service_time_series():
    """Анализирует обработанные временные ряды на предмет наличия данных в тестовом периоде."""
    print("🔍 Анализ service_time_series.pkl...")
    
    # Загружаем обработанные данные
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        service_time_series = pickle.load(f)
    
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = test_start + pd.offsets.MonthEnd(0)  # Последний день августа 2025

    print(f"Тестовый период: с {test_start.date()} по {test_end.date()}")
    print("=" * 80)

    for modality, ts in service_time_series.items():
        print(f"\n📊 Модальность: {modality}")
        
        # Универсальная обработка: всегда создаем DataFrame с колонками ['ds', 'y']
        if isinstance(ts, pd.Series):
            df = ts.reset_index()
            df.columns = ['ds', 'y']
        elif isinstance(ts, pd.DataFrame):
            if len(ts.columns) >= 2:
                df = ts.iloc[:, :2].copy()
                df.columns = ['ds', 'y']
            else:
                df = ts.reset_index()
                df.columns = ['ds', 'y']
        else:
            print(f"❌ Неподдерживаемый тип данных: {type(ts)}")
            continue

        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce').clip(lower=0)
        df = df.dropna(subset=['y']).reset_index(drop=True)

        # Фильтруем данные за тестовый период
        test_df = df[(df['ds'] >= test_start) & (df['ds'] <= test_end)].copy()

        print(f"   Всего дней в тестовом периоде: {len(test_df)}")
        print(f"   Сумма исследований в тестовом периоде: {test_df['y'].sum():,}")
        print(f"   Минимальное значение: {test_df['y'].min():,}")
        print(f"   Максимальное значение: {test_df['y'].max():,}")
        print(f"   Количество дней с нулевыми значениями: {(test_df['y'] == 0).sum()}")

        # Выводим первые 5 строк тестового периода
        if len(test_df) > 0:
            print("   Первые 5 дней тестового периода:")
            print(test_df.head().to_string(index=False))

    print("\n✅ Анализ завершен.")

if __name__ == "__main__":
    analyze_service_time_series()